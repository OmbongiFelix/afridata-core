"""
Ranking module — post-fusion ordering for the recommendations pipeline.

Receives a list of ScoredCandidate objects from hybrid.py and returns
a RankedList sorted by S_hybrid descending, trimmed to Top-N.

Optional diversity re-ranking:
  When EngineConfig.diversity_weight > 0, applies a Maximal Marginal
  Relevance (MMR) variant that penalises consecutive items from the
  same dataset category to improve result variety.

This module owns ALL post-fusion ordering logic.
hybrid.py must not sort or filter — it calls rank() from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Default number of results to return.  None means return all candidates.
DEFAULT_TOP_N: Optional[int] = 20

# When diversity re-ranking is active, items from the same category as a
# recently selected item have their score penalised by this factor.
DEFAULT_MMR_PENALTY: float = 0.5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(order=False)
class ScoredCandidate:
    """
    A single dataset candidate with its normalised hybrid score.

    Attributes
    ----------
    item_id:
        Unique dataset identifier.
    score:
        Normalised hybrid score in [0.0, 1.0] produced by hybrid.py.
    category:
        Optional category label used for MMR diversity re-ranking.
        When supplied, consecutive items from the same category are
        penalised.  Leave as ``None`` if category information is not
        available or diversity re-ranking is disabled.
    """

    item_id: int
    score: float
    category: Optional[str] = None

    def __repr__(self) -> str:
        cat = f", category={self.category!r}" if self.category else ""
        return f"ScoredCandidate(item_id={self.item_id}, score={self.score:.4f}{cat})"


# RankedList is a plain ordered list of ScoredCandidates.
# It is a type alias so callers can annotate return types clearly without
# importing a concrete wrapper class.
RankedList = list[ScoredCandidate]


# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------


@dataclass
class RankingConfig:
    """
    Optional configuration forwarded from the caller to ``rank()``.

    Attributes
    ----------
    top_n:
        Maximum number of results to return.  ``None`` returns all
        candidates in ranked order.  Defaults to ``DEFAULT_TOP_N``.
    diversity_weight:
        Weight in [0.0, 1.0] for MMR diversity re-ranking.
        0.0 disables diversity re-ranking entirely (pure score order).
        1.0 maximises diversity at the expense of relevance.
        Defaults to 0.0.
    mmr_penalty:
        Score penalty multiplier applied to candidates whose category
        matches a recently selected item.  Values in (0.0, 1.0) reduce
        score; 0.0 would permanently suppress an item.
        Only used when ``diversity_weight > 0``.
        Defaults to ``DEFAULT_MMR_PENALTY``.
    """

    top_n: Optional[int] = DEFAULT_TOP_N
    diversity_weight: float = 0.0
    mmr_penalty: float = DEFAULT_MMR_PENALTY

    def __post_init__(self) -> None:
        if self.top_n is not None and self.top_n < 1:
            raise ValueError(
                f"RankingConfig.top_n must be >= 1 or None, got {self.top_n}."
            )
        if not (0.0 <= self.diversity_weight <= 1.0):
            raise ValueError(
                f"RankingConfig.diversity_weight must be in [0.0, 1.0], "
                f"got {self.diversity_weight}."
            )
        if not (0.0 < self.mmr_penalty <= 1.0):
            raise ValueError(
                f"RankingConfig.mmr_penalty must be in (0.0, 1.0], "
                f"got {self.mmr_penalty}."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_by_score(candidates: list[ScoredCandidate]) -> RankedList:
    """
    Return candidates sorted by score descending (pure relevance order).

    Parameters
    ----------
    candidates:
        Unsorted list of ScoredCandidate objects.

    Returns
    -------
    RankedList
        New list sorted by ``score`` descending.  The input list is not
        mutated.
    """
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def _mmr_rerank(
    candidates: list[ScoredCandidate],
    diversity_weight: float,
    mmr_penalty: float,
) -> RankedList:
    """
    Apply a category-aware Maximal Marginal Relevance (MMR) variant to
    reduce consecutive items from the same dataset category.

    Algorithm
    ---------
    At each step:
      1. For every remaining candidate compute an *effective* score:

             effective(c) = (1 − λ) · score(c)  −  λ · penalty(c)

         where λ = ``diversity_weight`` and

             penalty(c) = mmr_penalty  if c.category is in selected_categories
                          else 0.0

      2. Select the candidate with the highest effective score.
      3. Record its category in ``selected_categories``.
      4. Repeat until no candidates remain.

    When ``diversity_weight == 0`` this degenerates to pure score order.
    When all categories are ``None``, no penalty is ever applied so the
    result is also equivalent to pure score order.

    Parameters
    ----------
    candidates:
        Unsorted list of ScoredCandidate objects.
    diversity_weight:
        λ in the MMR formula.  Must be in [0.0, 1.0].
    mmr_penalty:
        Penalty subtracted from the score of a candidate whose category
        is already represented in the selected set.

    Returns
    -------
    RankedList
        Diversity-re-ranked list of all input candidates.
    """
    remaining: list[ScoredCandidate] = list(candidates)
    selected: RankedList = []
    selected_categories: set[str] = set()

    lambda_ = diversity_weight
    relevance_weight = 1.0 - lambda_

    while remaining:
        best: Optional[ScoredCandidate] = None
        best_effective: float = float("-inf")

        for candidate in remaining:
            category_penalty = (
                mmr_penalty
                if (candidate.category is not None and candidate.category in selected_categories)
                else 0.0
            )
            effective = relevance_weight * candidate.score - lambda_ * category_penalty

            if effective > best_effective:
                best_effective = effective
                best = candidate

        # best is always set because remaining is non-empty
        assert best is not None
        selected.append(best)
        remaining.remove(best)

        if best.category is not None:
            selected_categories.add(best.category)

    return selected


def _apply_top_n(ranked: RankedList, top_n: Optional[int]) -> RankedList:
    """
    Trim a ranked list to at most ``top_n`` items.

    Parameters
    ----------
    ranked:
        Already-ordered RankedList.
    top_n:
        Maximum number of items to keep.  ``None`` returns the full list.

    Returns
    -------
    RankedList
        Trimmed (or unchanged) list.
    """
    if top_n is None:
        return ranked
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank(
    scored_candidates: list[ScoredCandidate],
    user_id: int,
    config: Optional[RankingConfig] = None,
) -> RankedList:
    """
    Order ``scored_candidates`` and return a trimmed ``RankedList``.

    This is the sole public entry point for ranking.  It is called by
    ``WeightedHybridEngine.recommend()`` in hybrid.py after scores have
    been fused and normalised.

    Steps
    -----
    1. Use default ``RankingConfig`` if none is supplied.
    2. Short-circuit and return ``[]`` if the input is empty.
    3. If ``diversity_weight > 0``, apply MMR re-ranking.
       Otherwise sort by score descending (faster, no penalty logic).
    4. Trim to ``top_n`` and return.

    Parameters
    ----------
    scored_candidates:
        Unsorted list of ``ScoredCandidate`` objects produced by
        ``hybrid._build_scored_candidates()``.
    user_id:
        The requesting user's ID.  Used only for structured logging.
    config:
        Optional ``RankingConfig``.  Defaults to ``RankingConfig()``
        (``top_n=20``, no diversity re-ranking).

    Returns
    -------
    RankedList
        Ordered, trimmed list of ``ScoredCandidate`` objects.

    Examples
    --------
    >>> candidates = [
    ...     ScoredCandidate(item_id=1, score=0.9, category="geo"),
    ...     ScoredCandidate(item_id=2, score=0.85, category="geo"),
    ...     ScoredCandidate(item_id=3, score=0.8, category="health"),
    ... ]
    >>> result = rank(candidates, user_id=42)
    >>> [c.item_id for c in result]
    [1, 2, 3]

    With diversity re-ranking (λ=0.4, penalty=0.5):
    >>> cfg = RankingConfig(diversity_weight=0.4, mmr_penalty=0.5, top_n=3)
    >>> result = rank(candidates, user_id=42, config=cfg)
    # item 2 (geo) is penalised after item 1 (geo) is selected, so
    # item 3 (health) may leapfrog it depending on effective scores.
    """
    if config is None:
        config = RankingConfig()

    if not scored_candidates:
        logger.debug("ranking.rank: empty candidate list for user_id=%d", user_id)
        return []

    logger.debug(
        "ranking.rank: user_id=%d, n_candidates=%d, top_n=%s, diversity_weight=%.3f",
        user_id,
        len(scored_candidates),
        config.top_n,
        config.diversity_weight,
    )

    # --- step 3: order candidates ----------------------------------------
    if config.diversity_weight > 0.0:
        ranked = _mmr_rerank(
            candidates=scored_candidates,
            diversity_weight=config.diversity_weight,
            mmr_penalty=config.mmr_penalty,
        )
        logger.debug(
            "ranking.rank: MMR re-ranking applied (λ=%.3f, penalty=%.3f)",
            config.diversity_weight,
            config.mmr_penalty,
        )
    else:
        ranked = _sort_by_score(scored_candidates)

    # --- step 4: trim to Top-N -------------------------------------------
    ranked = _apply_top_n(ranked, config.top_n)

    logger.info(
        "ranking.rank: user_id=%d → returning %d ranked items",
        user_id,
        len(ranked),
    )

    return ranked