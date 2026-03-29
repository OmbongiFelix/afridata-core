"""
Offline evaluation metrics for the recommendations domain.

Used by management commands and CI pipelines to measure model quality
against a held-out test set after every retraining run.
Not called in the live request path.

Functions:
  precision_at_k(recommended, relevant, k) -> float
  recall_at_k(recommended, relevant, k)    -> float
  ndcg_at_k(recommended, relevant, k)      -> float
  evaluate_engine(engine, test_interactions, k=10) -> dict
    Runs all three metrics and returns a summary dict.
"""

from __future__ import annotations

import logging
import math
from typing import List, Protocol, runtime_checkable

from recommendations.domain.schemas import RankedList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RecommendationEngine(Protocol):
    """
    Structural protocol satisfied by any engine that exposes a
    ``recommend()`` method compatible with this evaluation harness.

    The engine is expected to accept a ``user_id`` and return a
    ``RankedList`` (ordered list of item IDs).

    Using a Protocol rather than an abstract base class keeps the
    evaluation module decoupled from any concrete engine implementation —
    any class with a matching ``recommend`` signature satisfies this
    contract automatically.
    """

    def recommend(self, user_id: int) -> RankedList:
        ...


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def precision_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Fraction of the top-k recommended items that are relevant.

    Computed as: len(recommended[:k] ∩ relevant) / k

    Parameters
    ----------
    recommended:
        Ordered list of item IDs as returned by the recommendation engine.
    relevant:
        List of item IDs considered relevant for this user.
    k:
        Cut-off rank.  Only the first ``k`` items in ``recommended``
        are considered.

    Returns
    -------
    float
        P@k in [0.0, 1.0].  Returns 0.0 when ``k == 0``,
        ``recommended`` is empty, or ``k > len(recommended)``.

    Examples
    --------
    >>> precision_at_k([0, 1, 2, 3, 4], relevant=[0, 2, 4], k=3)
    0.6666666666666666
    """
    if k <= 0 or not recommended:
        return 0.0

    # Guard: clamp k to available recommendations rather than dividing by a
    # larger k than we have items — keeps the denominator honest.
    k = min(k, len(recommended))

    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item_id in top_k if item_id in relevant_set)
    return hits / k


def recall_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Fraction of all relevant items that appear in the top-k recommendations.

    Computed as: len(recommended[:k] ∩ relevant) / len(relevant)

    Parameters
    ----------
    recommended:
        Ordered list of item IDs as returned by the recommendation engine.
    relevant:
        List of item IDs considered relevant for this user.
    k:
        Cut-off rank.

    Returns
    -------
    float
        R@k in [0.0, 1.0].  Returns 0.0 when ``relevant`` is empty
        or ``k == 0``.

    Examples
    --------
    >>> recall_at_k([0, 1, 2, 3, 4], relevant=[0, 2, 4], k=3)
    0.6666666666666666
    """
    if k <= 0 or not relevant or not recommended:
        return 0.0

    k = min(k, len(recommended))
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item_id in top_k if item_id in relevant_set)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain at rank k.

    Uses binary relevance (1 if item ID is in ``relevant``, else 0).
    The ideal DCG (IDCG) is computed by assuming the maximum possible
    number of relevant items are placed at the top ranks.

    Computed as: DCG@k / IDCG@k  where

        DCG@k  = Σ rel(i) / log2(i + 2)   for i in 0 … k-1  (0-based rank)
        IDCG@k = Σ 1      / log2(i + 2)   for i in 0 … min(|relevant|, k)-1

    Parameters
    ----------
    recommended:
        Ordered list of item IDs as returned by the recommendation engine.
    relevant:
        List of item IDs considered relevant for this user.
    k:
        Cut-off rank.

    Returns
    -------
    float
        nDCG@k in [0.0, 1.0].  Returns 0.0 when ``relevant`` is empty,
        ``k == 0``, or no relevant items appear in the top-k.

    Examples
    --------
    >>> ndcg_at_k([0, 1, 2, 3, 4], relevant=[0, 1], k=3)
    1.0
    """
    if k <= 0 or not relevant or not recommended:
        return 0.0

    k = min(k, len(recommended))
    top_k = recommended[:k]
    relevant_set = set(relevant)

    # --- actual DCG --------------------------------------------------------
    dcg = sum(
        1.0 / math.log2(rank + 2)       # rank is 0-based → position is rank+1
        for rank, item_id in enumerate(top_k)
        if item_id in relevant_set
    )

    # --- ideal DCG (IDCG) --------------------------------------------------
    # Best case: min(|relevant|, k) hits placed at positions 0 … k-1.
    n_ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregated evaluation runner
# ---------------------------------------------------------------------------


def evaluate_engine(
    engine: RecommendationEngine,
    test_interactions: dict[int, set[int]],
    k: int = 10,
) -> dict:
    """
    Run all three metrics against ``engine`` over every user in
    ``test_interactions`` and return a summary dict.

    For each user, ``engine.recommend(user_id)`` is called and the result
    is evaluated against that user's held-out relevant item set using
    ``precision_at_k``, ``recall_at_k``, and ``ndcg_at_k``.
    Macro-averages (mean over users) are reported.

    Parameters
    ----------
    engine:
        Any object satisfying the ``RecommendationEngine`` protocol —
        i.e. any class with a ``recommend(user_id: int) -> RankedList``
        method.
    test_interactions:
        Mapping of ``user_id → set[item_id]`` for the held-out test set.
        Users with an empty relevant set are skipped with a warning.
    k:
        Cut-off rank applied to all three metrics.  Defaults to 10.

    Returns
    -------
    dict
        Keys:

        ``"precision"``  – macro-average P@k across evaluated users.
        ``"recall"``     – macro-average R@k across evaluated users.
        ``"ndcg"``       – macro-average nDCG@k across evaluated users.
        ``"k"``          – the cut-off rank used.
        ``"n_users"``    – number of users included in the averages.

    Example return value::

        {
            "precision": 0.43,
            "recall":    0.31,
            "ndcg":      0.52,
            "k":         10,
            "n_users":   512,
        }

    Raises
    ------
    ValueError
        If ``test_interactions`` is empty.
    """
    if not test_interactions:
        raise ValueError("test_interactions must not be empty.")

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    for user_id, relevant in test_interactions.items():
        if not relevant:
            logger.warning(
                "evaluate_engine: user_id=%d has an empty relevant set — skipping.",
                user_id,
            )
            continue

        try:
            ranked: RankedList = engine.recommend(user_id)
        except Exception:
            logger.exception(
                "evaluate_engine: engine.recommend() raised for user_id=%d — skipping.",
                user_id,
            )
            continue

        relevant_list = list(relevant)
        precisions.append(precision_at_k(ranked, relevant_list, k))
        recalls.append(recall_at_k(ranked, relevant_list, k))
        ndcgs.append(ndcg_at_k(ranked, relevant_list, k))

    n_users = len(precisions)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    avg_p = _mean(precisions)
    avg_r = _mean(recalls)
    avg_n = _mean(ndcgs)

    logger.info(
        "evaluate_engine: evaluated %d users at k=%d  "
        "P@k=%.4f  R@k=%.4f  nDCG@k=%.4f",
        n_users,
        k,
        avg_p,
        avg_r,
        avg_n,
    )

    return {
        "precision": avg_p,
        "recall":    avg_r,
        "ndcg":      avg_n,
        "k":         k,
        "n_users":   n_users,
    }