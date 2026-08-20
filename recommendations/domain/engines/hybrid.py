"""
Weighted Hybrid Fusion Engine — central orchestrator of the pipeline.

Formula:  S_hybrid = α · S_CF  +  (1 − α) · S_CBF

Orchestration sequence:
  1. Accept a CandidateSet and EngineConfig
  2. Call CollaborativeEngine.score() → S_CF dict
  3. Call ContentBasedEngine.score()  → S_CBF dict
  4. Fuse both dicts using the alpha formula
  5. Normalise fused scores to [0, 1]
  6. Pass ScoredCandidate list to domain/ranking.py
  7. Return the RankedList from ranking.rank()

This module does NOT sort, filter, or apply Top-N cutoff.
All post-fusion ordering belongs in domain/ranking.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from recommendations.domain.engines.collaborative import CollaborativeEngine
from recommendations.domain.engines.content_based import ContentBasedEngine
from recommendations.domain.schemas import CandidateSet, RankedList, ScoredCandidate
from recommendations.domain import ranking

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Default blend weight for collaborative filtering scores.
# α=0.5 gives equal weight to CF and CBF.
# Set closer to 1.0 to favour CF; closer to 0.0 to favour CBF.
DEFAULT_ALPHA: float = 0.5

# When a cold-start user is detected (all S_CF == 0.0), alpha is forced
# to 0.0 so the hybrid falls back entirely to content-based scores.
COLD_START_ALPHA: float = 0.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HybridEngineError(RuntimeError):
    """Raised for unrecoverable errors in the hybrid fusion engine."""


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """
    Runtime configuration for the WeightedHybridEngine.

    Attributes
    ----------
    alpha:
        Blend weight for collaborative filtering scores (S_CF).
        Must be in [0.0, 1.0].  The content-based weight is (1 - alpha).
        Defaults to ``DEFAULT_ALPHA`` (0.5).
    item_id_to_index:
        Mapping of dataset_id → row index in the collaborative model's
        item factor matrix.  Required by CollaborativeEngine.
    item_popularities:
        Mapping of dataset_id → popularity count.  Used by
        ContentBasedEngine for cold-start fallback scoring.
    interacted_item_ids:
        Ordered list of dataset IDs the user has interacted with.
        Used to build the CBF user profile vector.
    interaction_weights:
        Parallel list of weights for each interaction in
        ``interacted_item_ids``.  Use the WEIGHT_* constants from
        content_based.py (WEIGHT_DOWNLOAD, WEIGHT_VIEW, WEIGHT_IMPLICIT).
    auto_cold_start:
        If ``True`` (default), override alpha to ``COLD_START_ALPHA``
        when the CF engine returns all-zero scores (cold-start user).
    """

    alpha: float = DEFAULT_ALPHA
    item_id_to_index: dict[int, int] = field(default_factory=dict)
    item_popularities: dict[int, float] = field(default_factory=dict)
    interacted_item_ids: list[int] = field(default_factory=list)
    interaction_weights: list[float] = field(default_factory=list)
    auto_cold_start: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"EngineConfig.alpha must be in [0.0, 1.0], got {self.alpha}."
            )
        if len(self.interacted_item_ids) != len(self.interaction_weights):
            raise ValueError(
                "interacted_item_ids and interaction_weights must have the same length."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WeightedHybridEngine:
    """
    Orchestrates collaborative and content-based engines into a single
    weighted hybrid recommendation pipeline.
    """

    def __init__(
        self,
        collaborative_engine: Optional[CollaborativeEngine] = None,
        content_based_engine: Optional[ContentBasedEngine] = None,
        config: Optional[EngineConfig] = None,
        user: Any = None,
    ) -> None:
        self._cf = collaborative_engine if collaborative_engine is not None else CollaborativeEngine()
        self._cbf = content_based_engine if content_based_engine is not None else ContentBasedEngine()
        self.config = config or EngineConfig()
        self.user = user

    def fuse(
        self,
        cf_scores: dict[int, float],
        cbf_scores: dict[int, float],
        alpha: float,
        categories: Optional[dict[int, str]] = None,
    ) -> list[ScoredCandidate]:
        """
        Apply weighted fusion and return a normalised ScoredCandidate list.
        """
        all_ids = set(cf_scores) | set(cbf_scores)
        beta = 1.0 - alpha
        cats = categories or {}

        fused: dict[int, float] = {
            item_id: alpha * cf_scores.get(item_id, 0.0) + beta * cbf_scores.get(item_id, 0.0)
            for item_id in all_ids
        }

        normalised = _minmax_normalise(fused)
        return [
            ScoredCandidate(
                item_id=k,
                score=v,
                s_cf=cf_scores.get(k, 0.0),
                s_cbf=cbf_scores.get(k, 0.0),
                s_hybrid=v,
                category=cats.get(k, ""),
            )
            for k, v in normalised.items()
        ]

    def recommend(
        self,
        candidate_set: Optional[CandidateSet] = None,
        config: Optional[EngineConfig] = None,
        user_id: Optional[int] = None,
        strategy: str = "hybrid",
    ) -> RankedList:
        """
        Run recommendation pipeline for a single user and return a ranked list.
        """
        cfg = config or self.config or EngineConfig()

        if user_id is None:
            if candidate_set is not None:
                user_id = candidate_set.user_id
            elif self.user is not None:
                user_id = getattr(self.user, "id", getattr(self.user, "pk", 0))
            else:
                user_id = 1

        if candidate_set is None:
            from recommendations.domain.engines.candidate_generation import (
                CandidateGenerator,
            )
            candidate_set = CandidateGenerator().generate(user_id=user_id, config=cfg)

        candidate_ids = candidate_set.candidate_ids
        if not candidate_ids:
            return RankedList(user_id=user_id, items=[])

        # Auto-load engines if not loaded
        if not self._cf.is_loaded:
            try:
                self._cf.load()
            except Exception:
                pass
        if not self._cbf.is_loaded:
            try:
                self._cbf.load()
            except Exception:
                pass

        strat = (strategy or getattr(cfg, "strategy", "hybrid")).lower()

        # Score with CollaborativeEngine
        s_cf: dict[int, float] = {}
        if strat in ("hybrid", "collaborative", "cf"):
            s_cf = self._cf.score(user_id=user_id, candidates=candidate_set)

        # Score with ContentBasedEngine
        s_cbf: dict[int, float] = {}
        if strat in ("hybrid", "content", "content_based", "cbf"):
            s_cbf = self._cbf.score(user_id=user_id, candidates=candidate_set)

        # Determine effective alpha
        if strat in ("collaborative", "cf"):
            effective_alpha = 1.0
        elif strat in ("content", "content_based", "cbf"):
            effective_alpha = 0.0
        else:
            effective_alpha = cfg.alpha
            if getattr(cfg, "auto_cold_start", True) and getattr(self._cf, "is_cold_start", lambda s: all(v == 0.0 for v in s.values()))(s_cf):
                effective_alpha = COLD_START_ALPHA

        scored_candidates: list[ScoredCandidate] = self.fuse(s_cf, s_cbf, effective_alpha)

        # Delegate ordering to ranking
        ranked_list: RankedList = ranking.rank(
            scored_candidates=scored_candidates,
            user_id=user_id,
            config=cfg,
        )
        ranked_list.engine_used = strat
        ranked_list.alpha = effective_alpha

        return ranked_list

    def get_recommendations(
        self,
        user: Any = None,
        user_id: Optional[int] = None,
        top_n: int = 10,
        alpha: float = 0.5,
        strategy: str = "hybrid",
    ) -> RankedList:
        """
        Convenience wrapper for views and API handlers.
        """
        uid = user_id
        if uid is None and user is not None:
            uid = getattr(user, "id", getattr(user, "pk", 1))
        elif uid is None and self.user is not None:
            uid = getattr(self.user, "id", getattr(self.user, "pk", 1))
        if uid is None:
            uid = 1

        cfg = EngineConfig(top_n=top_n, alpha=alpha)
        return self.recommend(user_id=uid, config=cfg, strategy=strategy)

    def _require_engines_loaded(self) -> None:
        if not self._cf.is_loaded:
            raise HybridEngineError(
                "CollaborativeEngine has not been loaded. Call engine.load() first."
            )
        if not self._cbf.is_loaded:
            raise HybridEngineError(
                "ContentBasedEngine has not been loaded. Call engine.load() first."
            )


# Alias for spec-compliance; WeightedHybridEngine is the canonical name.
HybridEngine = WeightedHybridEngine


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _minmax_normalise(scores: dict[int, float]) -> dict[int, float]:
    """
    Min-max normalise a ``{item_id: score}`` dict to [0.0, 1.0].

    If all scores are identical (including all-zero), every item maps to 0.0.

    Parameters
    ----------
    scores:
        Raw fused scores.

    Returns
    -------
    dict[int, float]
        Normalised scores in [0.0, 1.0].
    """
    if not scores:
        return {}

    min_val = min(scores.values())
    max_val = max(scores.values())

    if max_val == min_val:
        return {k: 0.0 for k in scores}

    span = max_val - min_val
    return {k: (v - min_val) / span for k, v in scores.items()}