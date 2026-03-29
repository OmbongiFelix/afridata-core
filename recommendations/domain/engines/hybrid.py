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

    Intended to be instantiated once per process (e.g. in AppConfig.ready)
    after both sub-engines have been loaded, then called on every
    recommendation request.

    Parameters
    ----------
    collaborative_engine:
        A loaded ``CollaborativeEngine`` instance.
    content_based_engine:
        A loaded ``ContentBasedEngine`` instance.

    Examples
    --------
    >>> cf_engine = CollaborativeEngine()
    >>> cf_engine.load()
    >>> cbf_engine = ContentBasedEngine()
    >>> cbf_engine.load()
    >>> hybrid = WeightedHybridEngine(cf_engine, cbf_engine)
    >>> config = EngineConfig(
    ...     alpha=0.6,
    ...     item_id_to_index={101: 0, 202: 1, 303: 2},
    ...     item_popularities={101: 50, 202: 30, 303: 20},
    ...     interacted_item_ids=[101],
    ...     interaction_weights=[3.0],
    ... )
    >>> candidate_set = CandidateSet(
    ...     user_id=42,
    ...     candidate_ids=[202, 303],
    ...     seen_ids={101},
    ...     is_cold_start=False,
    ...     total_pool_size=3,
    ... )
    >>> ranked = hybrid.recommend(candidate_set, config)
    """

    def __init__(
        self,
        collaborative_engine: CollaborativeEngine,
        content_based_engine: ContentBasedEngine,
    ) -> None:
        self._cf = collaborative_engine
        self._cbf = content_based_engine

    def fuse(
        self,
        cf_scores: dict[int, float],
        cbf_scores: dict[int, float],
        alpha: float,
    ) -> list[ScoredCandidate]:
        """
        Apply weighted fusion and return a normalised ScoredCandidate list.

        Exposed as a public method so callers can invoke fusion directly
        with pre-computed scores (e.g. in tests or batch pipelines) without
        going through the full recommend() orchestration.

        Formula: S_hybrid[i] = α · S_CF[i] + (1 − α) · S_CBF[i]

        Items present in one dict but absent in the other are treated as 0.0.
        Fused scores are min-max normalised to [0.0, 1.0] before wrapping.

        Parameters
        ----------
        cf_scores:
            Collaborative filtering scores, mapping dataset_id → score.
        cbf_scores:
            Content-based filtering scores, mapping dataset_id → score.
        alpha:
            Blend weight for S_CF.  Must be in [0.0, 1.0].
            alpha=1.0 → CF only;  alpha=0.0 → CBF only.

        Returns
        -------
        list[ScoredCandidate]
            Unsorted; ordering is delegated to ranking.rank().
        """
        all_ids = set(cf_scores) | set(cbf_scores)
        beta = 1.0 - alpha

        fused: dict[int, float] = {
            item_id: alpha * cf_scores.get(item_id, 0.0) + beta * cbf_scores.get(item_id, 0.0)
            for item_id in all_ids
        }

        normalised = _minmax_normalise(fused)
        return [ScoredCandidate(item_id=k, score=v) for k, v in normalised.items()]

    def recommend(
        self,
        candidate_set: CandidateSet,
        config: EngineConfig,
    ) -> RankedList:
        """
        Run the full hybrid pipeline for a single user and return a
        ranked list of recommendations.

        Steps
        -----
        1. Validate that both sub-engines are loaded.
        2. Score candidates with CollaborativeEngine  → S_CF.
        3. Score candidates with ContentBasedEngine   → S_CBF.
        4. Detect cold-start and adjust alpha if needed.
        5. Fuse S_CF and S_CBF using the alpha formula.
        6. Normalise fused scores to [0, 1].
        7. Build ScoredCandidate list and delegate to ranking.rank().
        8. Return the RankedList.

        Parameters
        ----------
        candidate_set:
            Output of CandidateGenerator.generate().  Provides the list
            of unseen dataset IDs and the requesting user's ID.
        config:
            Runtime configuration including alpha, index mappings, and
            interaction history for profile construction.

        Returns
        -------
        RankedList
            Ordered recommendations from domain/ranking.rank().
            This engine does NOT apply Top-N cutoff; that is the
            responsibility of ranking.rank() or the calling view.

        Raises
        ------
        HybridEngineError
            If either sub-engine has not been loaded.
        """
        self._require_engines_loaded()

        user_id = candidate_set.user_id
        candidate_ids = candidate_set.candidate_ids

        logger.info(
            "hybrid.recommend: user_id=%d, n_candidates=%d, alpha=%.3f",
            user_id,
            len(candidate_ids),
            config.alpha,
        )

        # ---- step 2: collaborative filtering scores ---------------------
        s_cf: dict[int, float] = self._cf.score_for_user(
            user_id=user_id,
            candidate_item_ids=candidate_ids,
            item_id_to_index=config.item_id_to_index,
        )

        # ---- step 3: content-based filtering scores ---------------------
        s_cbf: dict[int, float] = self._cbf.score_for_user(
            interacted_item_ids=config.interacted_item_ids,
            interaction_weights=config.interaction_weights,
            candidate_item_ids=candidate_ids,
            item_popularities=config.item_popularities,
        )

        # ---- step 4: cold-start detection / alpha adjustment ------------
        effective_alpha = config.alpha
        if config.auto_cold_start and self._cf.is_cold_start(s_cf):
            effective_alpha = COLD_START_ALPHA
            logger.info(
                "hybrid.recommend: cold-start detected for user_id=%d; "
                "alpha overridden to %.1f",
                user_id,
                COLD_START_ALPHA,
            )

        # ---- steps 5-7: fuse, normalise, and build ScoredCandidates ----
        scored_candidates: list[ScoredCandidate] = self.fuse(s_cf, s_cbf, effective_alpha)

        logger.debug(
            "hybrid.recommend: fused %d scores with alpha=%.3f",
            len(scored_candidates),
            effective_alpha,
        )

        # ---- step 8: delegate ordering to ranking -----------------------
        ranked_list: RankedList = ranking.rank(
            scored_candidates=scored_candidates,
            user_id=user_id,
        )

        logger.info(
            "hybrid.recommend: user_id=%d, ranked %d items",
            user_id,
            len(ranked_list),
        )

        return ranked_list

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