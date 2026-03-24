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
from typing import Protocol, runtime_checkable

from ranking import RankedList, RankingConfig, ScoredCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RecommendationEngine(Protocol):
    """
    Structural protocol satisfied by any engine that exposes a
    ``recommend()`` method compatible with this evaluation harness.

    The engine is expected to return a ``RankedList`` for a given
    ``user_id``, optionally accepting a ``RankingConfig``.
    """

    def recommend(
        self,
        user_id: int,
        config: RankingConfig | None = None,
    ) -> RankedList:
        ...


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def precision_at_k(
    recommended: RankedList,
    relevant: set[int],
    k: int,
) -> float:
    """
    Fraction of the top-k recommended items that are relevant.

    Parameters
    ----------
    recommended:
        Ordered ``RankedList`` as returned by ``ranking.rank()``.
    relevant:
        Set of ``item_id`` values considered relevant for this user.
    k:
        Cut-off rank.  Only the first ``k`` items in ``recommended``
        are considered.

    Returns
    -------
    float
        P@k in [0.0, 1.0].  Returns 0.0 when ``k == 0`` or
        ``recommended`` is empty.

    Examples
    --------
    >>> candidates = [ScoredCandidate(i, score=1.0 - i * 0.1) for i in range(5)]
    >>> precision_at_k(candidates, relevant={0, 2, 4}, k=3)
    0.6666666666666666
    """
    if k <= 0:
        return 0.0

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for c in top_k if c.item_id in relevant)
    return hits / k


def recall_at_k(
    recommended: RankedList,
    relevant: set[int],
    k: int,
) -> float:
    """
    Fraction of all relevant items that appear in the top-k recommendations.

    Parameters
    ----------
    recommended:
        Ordered ``RankedList`` as returned by ``ranking.rank()``.
    relevant:
        Set of ``item_id`` values considered relevant for this user.
    k:
        Cut-off rank.

    Returns
    -------
    float
        R@k in [0.0, 1.0].  Returns 0.0 when ``relevant`` is empty
        or ``k == 0``.

    Examples
    --------
    >>> candidates = [ScoredCandidate(i, score=1.0 - i * 0.1) for i in range(5)]
    >>> recall_at_k(candidates, relevant={0, 2, 4}, k=3)
    0.6666666666666666
    """
    if k <= 0 or not relevant:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for c in top_k if c.item_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: RankedList,
    relevant: set[int],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain at rank k.

    Uses binary relevance (1 if ``item_id`` is in ``relevant``, else 0).
    The ideal DCG (IDCG) is computed by assuming the maximum possible
    number of relevant items are placed at the top ranks.

    Parameters
    ----------
    recommended:
        Ordered ``RankedList`` as returned by ``ranking.rank()``.
    relevant:
        Set of ``item_id`` values considered relevant for this user.
    k:
        Cut-off rank.

    Returns
    -------
    float
        nDCG@k in [0.0, 1.0].  Returns 0.0 when ``relevant`` is empty,
        ``k == 0``, or no relevant items appear in the top-k.

    Examples
    --------
    >>> candidates = [ScoredCandidate(i, score=1.0 - i * 0.1) for i in range(5)]
    >>> ndcg_at_k(candidates, relevant={0, 1}, k=3)
    1.0
    """
    if k <= 0 or not relevant:
        return 0.0

    top_k = recommended[:k]

    # --- actual DCG --------------------------------------------------------
    dcg = sum(
        1.0 / math.log2(rank + 2)          # rank is 0-based → log2(rank+2)
        for rank, candidate in enumerate(top_k)
        if candidate.item_id in relevant
    )

    # --- ideal DCG (IDCG) --------------------------------------------------
    # Best case: min(|relevant|, k) hits placed at positions 0 … k-1.
    n_ideal_hits = min(len(relevant), k)
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
    config: RankingConfig | None = None,
) -> dict:
    """
    Run all three metrics against ``engine`` over every user in
    ``test_interactions`` and return a summary dict.

    For each user, ``engine.recommend()`` is called and the resulting
    ``RankedList`` is evaluated against that user's held-out relevant
    item set using ``precision_at_k``, ``recall_at_k``, and ``ndcg_at_k``.
    Macro-averages (mean over users) are reported.

    Parameters
    ----------
    engine:
        Any object satisfying the ``RecommendationEngine`` protocol.
    test_interactions:
        Mapping of ``user_id → set[item_id]`` for the held-out test set.
        Users with an empty relevant set are skipped with a warning.
    k:
        Cut-off rank applied to all three metrics.  Defaults to 10.
    config:
        Optional ``RankingConfig`` forwarded to ``engine.recommend()``.
        When ``None``, the engine's own default config is used.

    Returns
    -------
    dict with keys:
        ``precision_at_k``  – macro-average P@k across evaluated users.
        ``recall_at_k``     – macro-average R@k across evaluated users.
        ``ndcg_at_k``       – macro-average nDCG@k across evaluated users.
        ``k``               – the cut-off rank used.
        ``n_users``         – number of users included in the averages.

    Example return value::

        {
            "precision_at_k": 0.43,
            "recall_at_k":    0.31,
            "ndcg_at_k":      0.52,
            "k":              10,
            "n_users":        512,
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
            ranked = engine.recommend(user_id, config=config)
        except Exception:
            logger.exception(
                "evaluate_engine: engine.recommend() raised for user_id=%d — skipping.",
                user_id,
            )
            continue

        precisions.append(precision_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    n_users = len(precisions)

    logger.info(
        "evaluate_engine: evaluated %d users at k=%d  "
        "P@k=%.4f  R@k=%.4f  nDCG@k=%.4f",
        n_users,
        k,
        sum(precisions) / n_users if n_users else 0.0,
        sum(recalls) / n_users if n_users else 0.0,
        sum(ndcgs) / n_users if n_users else 0.0,
    )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "precision_at_k": _mean(precisions),
        "recall_at_k": _mean(recalls),
        "ndcg_at_k": _mean(ndcgs),
        "k": k,
        "n_users": n_users,
    }