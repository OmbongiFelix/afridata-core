"""
Candidate generation engine for the recommendations pipeline.

Retrieves the set of dataset IDs that are eligible to be recommended
to a given user. Filters out items the user has already interacted
with so that collaborative.py and content_based.py only score
genuinely new candidates.

Responsibilities:
  1. Fetch all available dataset IDs via persistence.get_all_dataset_ids()
  2. Fetch the user's interaction history via persistence.get_user_interactions()
  3. Subtract seen items from the full pool
  4. Apply optional recency or popularity pre-filters to cap pool size
  5. Return a CandidateSet schema object
"""

from __future__ import annotations

import logging
from typing import Optional

from recommendations.domain.schemas import CandidateSet, EngineConfig
from recommendations.infrastructure.persistence import (
    get_all_dataset_ids,
    get_user_interactions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Fallback pool cap when EngineConfig.candidate_pool_size is not set.
# Prevents the scoring engines from receiving an unbounded item list on
# large catalogues.
DEFAULT_MAX_POOL_SIZE: int = 5_000

# When popularity-based pre-filtering is active, this is the minimum
# interaction_count a dataset must have to pass the filter.
DEFAULT_MIN_POPULARITY: int = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CandidateGenerationError(RuntimeError):
    """Raised for unrecoverable errors during candidate generation."""


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------


def _apply_popularity_filter(
    candidate_ids: list[int],
    item_popularities: dict[int, int],
    min_popularity: int,
) -> list[int]:
    """
    Remove datasets whose interaction_count is below *min_popularity*.

    Parameters
    ----------
    candidate_ids:
        Pool of unseen dataset IDs to filter.
    item_popularities:
        Mapping of dataset_id → interaction_count.
    min_popularity:
        Datasets with fewer interactions than this threshold are removed.

    Returns
    -------
    list[int]
        Filtered candidate list (same order as input).
    """
    filtered = [
        item_id
        for item_id in candidate_ids
        if item_popularities.get(item_id, 0) >= min_popularity
    ]
    logger.debug(
        "candidate_generation._apply_popularity_filter: "
        "%d → %d candidates (min_popularity=%d)",
        len(candidate_ids),
        len(filtered),
        min_popularity,
    )
    return filtered


def _apply_recency_filter(
    candidate_ids: list[int],
    item_recency_scores: dict[int, float],
    top_n: int,
) -> list[int]:
    """
    Keep only the *top_n* most recent datasets.

    Recency is determined by ``item_recency_scores``, a mapping of
    dataset_id → recency score (higher = more recent).  Any item not
    present in the mapping receives a recency score of 0.0.

    Parameters
    ----------
    candidate_ids:
        Pool of unseen dataset IDs.
    item_recency_scores:
        Mapping of dataset_id → recency score.
    top_n:
        Maximum number of candidates to return.

    Returns
    -------
    list[int]
        Up to *top_n* candidates ordered by descending recency score.
    """
    sorted_ids = sorted(
        candidate_ids,
        key=lambda item_id: item_recency_scores.get(item_id, 0.0),
        reverse=True,
    )
    result = sorted_ids[:top_n]
    logger.debug(
        "candidate_generation._apply_recency_filter: "
        "%d → %d candidates (top_n=%d)",
        len(candidate_ids),
        len(result),
        top_n,
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """
    Generates the pool of candidate dataset IDs eligible to be recommended
    to a given user.

    Intended to run at the start of every recommendation request, before
    the collaborative and content-based engines score individual items.
    This stage owns the seen-item filter — no downstream engine should
    duplicate that logic.

    Parameters
    ----------
    min_popularity:
        Datasets with fewer than this many interactions are removed from
        the candidate pool when ``config.apply_popularity_filter`` is
        ``True``.  Defaults to ``DEFAULT_MIN_POPULARITY``.

    Examples
    --------
    >>> from recommendations.domain.schemas import EngineConfig
    >>> generator = CandidateGenerator()
    >>> config = EngineConfig(candidate_pool_size=1000)
    >>> candidate_set = generator.generate(user_id=42, config=config)
    >>> len(candidate_set)
    987
    """

    def __init__(self, min_popularity: int = DEFAULT_MIN_POPULARITY) -> None:
        self._min_popularity = min_popularity

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        user_id: int,
        config: EngineConfig,
        item_popularities: Optional[dict[int, int]] = None,
        item_recency_scores: Optional[dict[int, float]] = None,
    ) -> CandidateSet:
        """
        Build a ``CandidateSet`` for *user_id*.

        Steps
        -----
        1. Fetch all active dataset IDs from persistence.
        2. Fetch the user's interaction history from persistence.
        3. Subtract seen items from the full pool.
        4. Optionally apply popularity and/or recency pre-filters
           (controlled by ``config``).
        5. Cap the pool at ``config.candidate_pool_size`` (most popular
           items kept; falls back to ``DEFAULT_MAX_POOL_SIZE``).
        6. Return a ``CandidateSet`` — always a valid object, never an
           error, even when all items have been seen (returns empty list).

        Parameters
        ----------
        user_id:
            Primary key of the requesting user.
        config:
            Engine configuration object.  The relevant fields are:
            ``candidate_pool_size`` — hard cap on pool size (``None``
            falls back to ``DEFAULT_MAX_POOL_SIZE``);
            ``apply_popularity_filter`` — if ``True`` and
            ``item_popularities`` is supplied, remove low-popularity items;
            ``apply_recency_filter`` — if ``True`` and
            ``item_recency_scores`` is supplied, restrict to most-recent items.
        item_popularities:
            Mapping of dataset_id → interaction_count.  Required when
            ``config.apply_popularity_filter`` is ``True`` or when the pool
            needs to be capped by popularity.  If ``None`` an empty dict
            is used (no popularity information available).
        item_recency_scores:
            Mapping of dataset_id → recency score (higher = more recent).
            Required when ``config.apply_recency_filter`` is ``True``.
            If ``None`` an empty dict is used.

        Returns
        -------
        CandidateSet
            Contains the filtered, capped list of candidate IDs together
            with metadata useful to downstream engines.  When the user has
            interacted with every active item, ``candidate_ids`` is an
            empty list — this is not an error condition.

        Raises
        ------
        CandidateGenerationError
            If the persistence layer raises an unexpected error.
        """
        popularities: dict[int, int] = item_popularities or {}
        recency_scores: dict[int, float] = item_recency_scores or {}
        max_pool_size: int = (
            config.candidate_pool_size
            if getattr(config, "candidate_pool_size", None)
            else DEFAULT_MAX_POOL_SIZE
        )

        # ---- step 1: full active pool -----------------------------------
        try:
            all_ids: list[int] = get_all_dataset_ids()
        except Exception as exc:
            raise CandidateGenerationError(
                f"Failed to fetch all dataset IDs for user_id={user_id}."
            ) from exc

        total_pool_size = len(all_ids)
        logger.info(
            "candidate_generation.generate: user_id=%d, total_pool=%d",
            user_id,
            total_pool_size,
        )

        # ---- step 2: interaction history --------------------------------
        try:
            interactions = get_user_interactions(user_id)
        except Exception as exc:
            raise CandidateGenerationError(
                f"Failed to fetch interactions for user_id={user_id}."
            ) from exc

        seen_ids: set[int] = {interaction.dataset_id for interaction in interactions}
        is_cold_start = len(seen_ids) == 0

        logger.info(
            "candidate_generation.generate: user_id=%d, seen=%d, cold_start=%s",
            user_id,
            len(seen_ids),
            is_cold_start,
        )

        # ---- step 3: subtract seen items --------------------------------
        # Returns an empty list (not an error) when the user has seen
        # every active item — callers must handle len(candidate_set) == 0.
        candidates: list[int] = [
            item_id for item_id in all_ids if item_id not in seen_ids
        ]

        logger.debug(
            "candidate_generation.generate: user_id=%d, unseen_pool=%d",
            user_id,
            len(candidates),
        )

        # ---- step 4a: optional popularity pre-filter --------------------
        if getattr(config, "apply_popularity_filter", False) and popularities:
            candidates = _apply_popularity_filter(
                candidate_ids=candidates,
                item_popularities=popularities,
                min_popularity=self._min_popularity,
            )

        # ---- step 4b: optional recency pre-filter -----------------------
        if getattr(config, "apply_recency_filter", False) and recency_scores:
            candidates = _apply_recency_filter(
                candidate_ids=candidates,
                item_recency_scores=recency_scores,
                top_n=max_pool_size,
            )

        # ---- step 5: hard cap by popularity -----------------------------
        if len(candidates) > max_pool_size:
            candidates = sorted(
                candidates,
                key=lambda item_id: popularities.get(item_id, 0),
                reverse=True,
            )[:max_pool_size]
            logger.debug(
                "candidate_generation.generate: pool capped at %d",
                max_pool_size,
            )

        logger.info(
            "candidate_generation.generate: user_id=%d, final_candidates=%d",
            user_id,
            len(candidates),
        )

        return CandidateSet(
            user_id=user_id,
            candidate_ids=candidates,
            seen_ids=seen_ids,
            is_cold_start=is_cold_start,
            total_pool_size=total_pool_size,
        )