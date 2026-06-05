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

from datetime import timedelta
from typing import TYPE_CHECKING

from django.contrib.auth import get_user_model
from django.db.models import QuerySet
from django.utils import timezone

from recommendations.models import (
    DatasetProxy,
    RecommendationResult,
    UserInteraction,
)

if TYPE_CHECKING:
    from recommendations.domain.schemas import RankedList


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recommendations older than this are considered stale and will not be
# returned by get_latest_recommendation(); callers should trigger a fresh
# engine run instead.
RECOMMENDATION_MAX_AGE = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_user_exists(user_pk: int) -> None:
    """
    Raise ValueError if no CustomUser row exists for the given primary key.

    Using get_user_model() keeps this decoupled from the concrete model
    import and respects AUTH_USER_MODEL (which is CustomUser here, with
    USERNAME_FIELD='email').  The PK is always the integer AutoField `id`,
    regardless of which field is used for login.

    Parameters
    ----------
    user_pk:
        Integer primary key of the CustomUser record.

    Raises
    ------
    ValueError
        If the user does not exist in the database.
    """
    User = get_user_model()
    if not User.objects.filter(pk=user_pk).exists():
        raise ValueError(
            f"No user found with pk={user_pk!r}. "
            "Pass the integer primary key (CustomUser.id), not the email."
        )


# ---------------------------------------------------------------------------
# User interactions
# ---------------------------------------------------------------------------


def get_user_interactions(user_pk: int) -> list[UserInteraction]:
    """
    Return all UserInteraction records for the given user, ordered by
    most recent first.

    Parameters
    ----------
    user_pk:
        Integer primary key (CustomUser.id) of the target user.
        Do NOT pass the email address — CustomUser uses email as
        USERNAME_FIELD for authentication, but the DB primary key
        is always the integer `id` field.

    Returns
    -------
    list[UserInteraction]
        May be empty if the user has no recorded interactions (cold-start).

    Raises
    ------
    ValueError
        If no CustomUser exists for user_pk.

    Notes
    -----
    select_related("user") has been removed.  Every row already belongs
    to the same user (we filter by user_pk), so the JOIN adds cost with
    no benefit.  Domain code that needs user attributes should accept the
    user object as a separate argument rather than traversing the FK.
    """
    _assert_user_exists(user_pk)

    return list(
        UserInteraction.objects.filter(user_id=user_pk).order_by("-created_at")
    )


# ---------------------------------------------------------------------------
# Dataset IDs
# ---------------------------------------------------------------------------


def get_all_dataset_ids() -> list[int]:
    """
    Return the primary keys of every active DatasetProxy record.

    Candidate generation uses this pool as the universe of items that
    can be recommended before seen-item filtering is applied.

    Returns
    -------
    list[int]
        Ordered by descending interaction_count (popularity), matching
        DatasetProxy.Meta.ordering.  May be empty if no datasets are
        synced yet.

    Notes
    -----
    Fix (field name): values_list now targets "id" (Django's AutoField PK
    on Dataset / DatasetProxy).  The previous "dataset_id" does not exist
    on the model and would raise a FieldError at runtime.

    Fix (is_active): the filter is kept here but DatasetProxy must define
    this field explicitly (Dataset itself has no is_active column).  If
    DatasetProxy does not add the field, remove the filter or replace it
    with whichever flag the proxy uses to mark inactive records.
    """
    return list(
        DatasetProxy.objects.filter(is_active=True).values_list("id", flat=True)
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

    Notes
    -----
    Same is_active caveat as get_all_dataset_ids() — the field must be
    defined on DatasetProxy (or Dataset) for this filter to work.
    """
    return DatasetProxy.objects.filter(is_active=True)


# ---------------------------------------------------------------------------
# Recommendation results
# ---------------------------------------------------------------------------


def save_recommendation_result(
    user_pk: int,
    ranked_list: "RankedList",
    alpha: float,
    engine_used: str = RecommendationResult.EngineUsed.HYBRID,
    candidate_pool_size: int = 0,
) -> RecommendationResult:
    """
    Persist (or update) the Top-N recommendation result for a user.

    Uses update_or_create on the OneToOne ``user`` field so that only
    one result row ever exists per user — calling this a second time
    *replaces* the previous result rather than inserting a duplicate.

    Parameters
    ----------
    user_pk:
        Integer primary key (CustomUser.id) of the target user.
    ranked_list:
        The RankedList produced by the hybrid engine.
        ``ranked_list.items`` must be a list of ScoredCandidate objects.
    alpha:
        The content-based weight used during fusion (0 = pure CF,
        1 = pure content-based).
    engine_used:
        One of the RecommendationResult.EngineUsed choices.
        Defaults to HYBRID.
    candidate_pool_size:
        Number of candidates evaluated before trimming to Top-N.

    Returns
    -------
    RecommendationResult
        The freshly saved (or updated) instance.

    Raises
    ------
    ValueError
        If no CustomUser exists for user_pk.

    Notes
    -----
    Fix (score extraction): s_hybrid may be None when engine_used is not
    HYBRID (pure CF or pure content-based runs populate only s_cf or
    s_content).  We now fall back through s_cf → s_content → 0.0 so the
    list is always populated with a valid float and never raises TypeError.
    """
    _assert_user_exists(user_pk)

    ranked_ids = [int(item.item_id) for item in ranked_list.items]

    # s_hybrid is only guaranteed to be present for HYBRID engine runs.
    # Fall back to the single-engine score that is available, then to 0.0.
    scores = [
        float(
            item.s_hybrid
            if item.s_hybrid is not None
            else getattr(item, "s_cf", None)
            if getattr(item, "s_cf", None) is not None
            else getattr(item, "s_content", None)
            if getattr(item, "s_content", None) is not None
            else 0.0
        )
        for item in ranked_list.items
    ]

    result, _ = RecommendationResult.objects.update_or_create(
        user_id=user_pk,
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


def get_latest_recommendation(user_pk: int) -> RecommendationResult | None:
    """
    Return the most recently persisted recommendation result for a user,
    or None if no result exists or the cached result has gone stale.

    The API layer calls this before falling back to a live engine run,
    so it must be fast.  The OneToOne relation means at most one row
    is read.

    Parameters
    ----------
    user_pk:
        Integer primary key (CustomUser.id) of the target user.

    Returns
    -------
    RecommendationResult | None
        None if no result has been stored yet, or if the stored result
        is older than RECOMMENDATION_MAX_AGE (24 hours by default).

    Raises
    ------
    ValueError
        If no CustomUser exists for user_pk.

    Notes
    -----
    Fix (staleness): the previous implementation returned whatever was
    cached regardless of age.  Recommendations generated days ago are no
    longer valid as user interactions accumulate.  We now compare
    generated_at against the current time and return None for stale
    results so that callers trigger a fresh engine run.
    """
    _assert_user_exists(user_pk)

    try:
        result = RecommendationResult.objects.get(user_id=user_pk)
    except RecommendationResult.DoesNotExist:
        return None

    if timezone.now() - result.generated_at > RECOMMENDATION_MAX_AGE:
        return None

    return result