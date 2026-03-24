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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from infrastructure.persistence import get_all_dataset_ids, get_user_interactions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Maximum candidate pool size returned when no explicit cap is given.
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
# Schema
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """
    Output schema for the candidate generation step.

    Attributes
    ----------
    user_id:
        The user for whom candidates were generated.
    candidate_ids:
        Ordered list of dataset IDs eligible for scoring.
        Items the user has already seen are excluded.
    seen_ids:
        Set of dataset IDs the user has previously interacted with
        (used downstream to enforce exclusion at scoring time).
    is_cold_start:
        True if the user has no recorded interactions.
    total_pool_size:
        Size of the full active-dataset pool before filtering.
    generated_at:
        UTC timestamp of candidate generation.
    """

    user_id: int
    candidate_ids: list[int]
    seen_ids: set[int]
    is_cold_start: bool
    total_pool_size: int
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def __len__(self) -> int:
        return len(self.candidate_ids)


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

    Parameters
    ----------
    max_pool_size:
        Hard cap on the number of candidates returned.  When the unseen
        pool exceeds this value the most popular items are kept.
        Defaults to ``DEFAULT_MAX_POOL_SIZE``.
    min_popularity:
        Datasets with fewer than this many interactions are removed from
        the candidate pool when ``apply_popularity_filter=True``.
        Defaults to ``DEFAULT_MIN_POPULARITY``.

    Examples
    --------
    >>> generator = CandidateGenerator(max_pool_size=1000)
    >>> candidate_set = generator.generate(user_id=42)
    >>> len(candidate_set)
    987
    """

    def __init__(
        self,
        max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
        min_popularity: int = DEFAULT_MIN_POPULARITY,
    ) -> None:
        self._max_pool_size = max_pool_size
        self._min_popularity = min_popularity

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        user_id: int,
        item_popularities: Optional[dict[int, int]] = None,
        item_recency_scores: Optional[dict[int, float]] = None,
        apply_popularity_filter: bool = False,
        apply_recency_filter: bool = False,
    ) -> CandidateSet:
        """
        Build a ``CandidateSet`` for *user_id*.

        Steps
        -----
        1. Fetch all active dataset IDs from persistence.
        2. Fetch the user's interaction history from persistence.
        3. Subtract seen items from the full pool.
        4. Optionally apply popularity and/or recency pre-filters.
        5. Cap the pool at ``max_pool_size`` (most popular items kept).
        6. Return a ``CandidateSet``.

        Parameters
        ----------
        user_id:
            Primary key of the requesting user.
        item_popularities:
            Mapping of dataset_id → interaction_count.  Required when
            ``apply_popularity_filter=True`` or when the pool needs to
            be capped by popularity.  If ``None`` an empty dict is used
            (no popularity information available).
        item_recency_scores:
            Mapping of dataset_id → recency score (higher = more recent).
            Required when ``apply_recency_filter=True``.  If ``None``
            an empty dict is used.
        apply_popularity_filter:
            If ``True``, remove datasets below ``min_popularity`` from
            the candidate pool before capping.
        apply_recency_filter:
            If ``True``, restrict the pool to the top-``max_pool_size``
            most recent items (applied after the popularity filter and
            before the hard size cap).

        Returns
        -------
        CandidateSet
            Contains the filtered, capped list of candidate IDs together
            with metadata useful to downstream engines.

        Raises
        ------
        CandidateGenerationError
            If the persistence layer raises an unexpected error.
        """
        popularities: dict[int, int] = item_popularities or {}
        recency_scores: dict[int, float] = item_recency_scores or {}

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
        candidates: list[int] = [
            item_id for item_id in all_ids if item_id not in seen_ids
        ]

        logger.debug(
            "candidate_generation.generate: user_id=%d, unseen_pool=%d",
            user_id,
            len(candidates),
        )

        # ---- step 4a: optional popularity pre-filter --------------------
        if apply_popularity_filter and popularities:
            candidates = _apply_popularity_filter(
                candidate_ids=candidates,
                item_popularities=popularities,
                min_popularity=self._min_popularity,
            )

        # ---- step 4b: optional recency pre-filter -----------------------
        if apply_recency_filter and recency_scores:
            candidates = _apply_recency_filter(
                candidate_ids=candidates,
                item_recency_scores=recency_scores,
                top_n=self._max_pool_size,
            )

        # ---- step 5: hard cap by popularity -----------------------------
        if len(candidates) > self._max_pool_size:
            candidates = sorted(
                candidates,
                key=lambda item_id: popularities.get(item_id, 0),
                reverse=True,
            )[: self._max_pool_size]
            logger.debug(
                "candidate_generation.generate: pool capped at %d",
                self._max_pool_size,
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