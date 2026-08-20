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
from datetime import datetime, timezone
from typing import Any, List, Optional

from recommendations.domain.schemas import (
    EngineConfig,
    RankedList,
    ScoredCandidate,
)

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
        matches a recently selected item.  Values in (0.0, 1.0] reduce
        score; 0.0 would permanently suppress an item and is therefore
        disallowed.
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
def _sort_by_score(candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
    """
    Return candidates sorted by score descending, breaking ties by
    ``item_id`` ascending for deterministic output.
    """
    return sorted(
        candidates,
        key=lambda c: (
            -float(getattr(c, "s_hybrid", getattr(c, "score", 0.0))),
            str(c.item_id),
        ),
    )


def _mmr_rerank(
    candidates: List[ScoredCandidate],
    diversity_weight: float,
    mmr_penalty: float,
) -> List[ScoredCandidate]:
    """
    Apply a category-aware Maximal Marginal Relevance (MMR) variant to
    reduce consecutive items from the same dataset category.
    """
    remaining: List[ScoredCandidate] = list(candidates)
    selected: List[ScoredCandidate] = []
    selected_categories: set[str] = set()

    lambda_ = diversity_weight
    relevance_weight = 1.0 - lambda_

    while remaining:
        best: Optional[ScoredCandidate] = None
        best_effective: float = float("-inf")

        for candidate in remaining:
            cand_cat = candidate.category if getattr(candidate, "category", None) else None
            category_penalty = (
                mmr_penalty
                if (
                    cand_cat is not None
                    and cand_cat in selected_categories
                )
                else 0.0
            )
            c_score = float(getattr(candidate, "s_hybrid", getattr(candidate, "score", 0.0)))
            effective = relevance_weight * c_score - lambda_ * category_penalty

            # Tie-break on item_id ascending for determinism
            if effective > best_effective or (
                effective == best_effective
                and best is not None
                and str(candidate.item_id) < str(best.item_id)
            ):
                best_effective = effective
                best = candidate

        assert best is not None
        selected.append(best)
        remaining.remove(best)

        if getattr(best, "category", None):
            selected_categories.add(best.category)

    return selected


def _apply_top_n(
    ranked: List[ScoredCandidate], top_n: Optional[int]
) -> List[ScoredCandidate]:
    if top_n is None:
        return ranked
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank(
    scored_candidates: List[ScoredCandidate],
    user_id: int | RankingConfig | EngineConfig = 0,
    config: Optional[RankingConfig | EngineConfig] = None,
) -> RankedList:
    """
    Order ``scored_candidates`` and return a trimmed ``RankedList``.
    """
    if isinstance(user_id, (RankingConfig, EngineConfig)):
        config = user_id
        user_id_int = 0
    else:
        user_id_int = int(user_id) if isinstance(user_id, int) else 0

    if config is None:
        cfg = RankingConfig()
    elif isinstance(config, EngineConfig):
        cfg = RankingConfig(
            top_n=config.top_n,
            diversity_weight=config.diversity_weight,
        )
    elif isinstance(config, RankingConfig):
        cfg = config
    else:
        cfg = RankingConfig()

    if not scored_candidates:
        logger.debug("ranking.rank: empty candidate list for user_id=%d", user_id_int)
        return RankedList(user_id=user_id_int)

    logger.debug(
        "ranking.rank: user_id=%d, n_candidates=%d, top_n=%s, diversity_weight=%.3f",
        user_id_int,
        len(scored_candidates),
        cfg.top_n,
        cfg.diversity_weight,
    )

    # --- step 3: order candidates ----------------------------------------
    if cfg.diversity_weight > 0.0:
        ordered = _mmr_rerank(
            candidates=scored_candidates,
            diversity_weight=cfg.diversity_weight,
            mmr_penalty=cfg.mmr_penalty,
        )
    else:
        ordered = _sort_by_score(scored_candidates)

    # --- step 4: trim to Top-N -------------------------------------------
    ordered = _apply_top_n(ordered, cfg.top_n)

    # --- step 5: wrap with metadata and return ---------------------------
    result = RankedList(user_id=user_id_int, items=ordered)

    logger.info(
        "ranking.rank: user_id=%d → returning %d ranked items (generated_at=%s)",
        user_id_int,
        len(result),
        result.generated_at.isoformat(),
    )

    return result