"""
Django ORM query helpers for the recommendations app.

This is the only module in the app that imports from models.py.
Domain code (engines, ranking) must never query the DB directly —
they call functions from here instead.

Functions:
  get_user_interactions(user_id) -> list[UserInteraction]
  get_all_dataset_ids()          -> list[int]
  get_all_datasets()             -> QuerySet[DatasetProxy]
  save_recommendation_result(user_id, ranked_list, alpha) -> RecommendationResult
  get_latest_recommendation(user_id) -> RecommendationResult | None
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models import QuerySet

from recommendations.models import (
    DatasetProxy,
    RecommendationResult,
    UserInteraction,
)

if TYPE_CHECKING:
    from recommendations.domain.schemas import RankedList


# ---------------------------------------------------------------------------
# User interactions
# ---------------------------------------------------------------------------


def get_user_interactions(user_id: int) -> list[UserInteraction]:
    """
    Return all UserInteraction records for the given user, ordered by
    most recent first.

    ``select_related("user")`` is applied so that any FK traversal on
    ``interaction.user`` in domain code does not trigger an extra query
    per row.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.

    Returns
    -------
    list[UserInteraction]
        May be empty if the user has no recorded interactions (cold-start).
    """
    return list(
        UserInteraction.objects.filter(user_id=user_id)
        .select_related("user")
        .order_by("-created_at")
    )


# ---------------------------------------------------------------------------
# Dataset IDs
# ---------------------------------------------------------------------------


def get_all_dataset_ids() -> list[int]:
    """
    Return the primary keys of every *active* DatasetProxy record.

    Candidate generation uses this pool as the universe of items that
    can be recommended before seen-item filtering is applied.

    Returns
    -------
    list[int]
        Ordered by descending interaction_count (popularity), matching
        DatasetProxy.Meta.ordering.  May be empty if no datasets are
        synced yet.
    """
    return list(
        DatasetProxy.objects.filter(is_active=True).values_list("dataset_id", flat=True)
    )


def get_all_datasets() -> QuerySet[DatasetProxy]:
    """
    Return a QuerySet of all active DatasetProxy objects.

    Used by the content-based training command to build the TF-IDF corpus.
    Returns a lazy QuerySet so callers can apply further filters or
    annotations without an extra round-trip.

    Returns
    -------
    QuerySet[DatasetProxy]
    """
    return DatasetProxy.objects.filter(is_active=True)


# ---------------------------------------------------------------------------
# Recommendation results
# ---------------------------------------------------------------------------


def save_recommendation_result(
    user_id: int,
    ranked_list: "RankedList",
    alpha: float,
    engine_used: str = RecommendationResult.EngineUsed.HYBRID,
    candidate_pool_size: int = 0,
) -> RecommendationResult:
    """
    Persist (or update) the Top-N recommendation result for a user.

    Uses ``update_or_create`` on the OneToOne ``user`` field so that only
    one result row ever exists per user — calling this a second time
    *replaces* the previous result rather than inserting a duplicate.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.
    ranked_list:
        The :class:`~recommendations.domain.schemas.RankedList` produced
        by the hybrid engine.  ``ranked_list.items`` must be a list of
        ``ScoredCandidate`` objects.
    alpha:
        The content-based weight used during fusion (0 = pure CF,
        1 = pure content-based).
    engine_used:
        One of the ``RecommendationResult.EngineUsed`` choices.
        Defaults to ``HYBRID``.
    candidate_pool_size:
        Number of candidates evaluated before trimming to Top-N.

    Returns
    -------
    RecommendationResult
        The freshly saved (or updated) instance.
    """
    ranked_ids = [int(item.item_id) for item in ranked_list.items]
    scores = [float(item.s_hybrid) for item in ranked_list.items]

    result, _ = RecommendationResult.objects.update_or_create(
        user_id=user_id,
        defaults={
            "ranked_dataset_ids": ranked_ids,
            "scores": scores,
            "alpha": alpha,
            "engine_used": engine_used,
            "candidate_pool_size": candidate_pool_size,
            "generated_at": ranked_list.generated_at,
        },
    )
    return result


def get_latest_recommendation(user_id: int) -> RecommendationResult | None:
    """
    Return the most recently persisted recommendation result for a user,
    or ``None`` if no result has been stored yet.

    The API layer calls this before falling back to a live engine run,
    so it must be fast.  The OneToOne relation means at most one row
    is read.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.

    Returns
    -------
    RecommendationResult | None
    """
    try:
        return RecommendationResult.objects.get(user_id=user_id)
    except RecommendationResult.DoesNotExist:
        return None