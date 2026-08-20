"""
Celery task definitions for the recommendations app.

Tasks:

  refresh_user_scores(user_id)
    Recomputes and caches Top-N recommendations for one user.
    Called by signals when a UserInteraction is created or deleted.

  train_collaborative_task()
    Full refit of the collaborative filter from interaction history.
    Triggered nightly via Celery beat or by the management command.

  train_content_based_task()
    Rebuilds the TF-IDF matrix from current Dataset metadata.
    Run after bulk dataset metadata updates.

All tasks must be idempotent and safe to retry on failure.
"""

import logging
import time

from celery import shared_task

from recommendations.domain.engines.hybrid import HybridEngine
from recommendations.infrastructure.cache import set_cached_recommendations

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def refresh_user_scores(self, user_id: int) -> None:
    """
    Recompute and cache Top-N recommendations for a single user.

    Called by signals when a UserInteraction is created or deleted.
    Delegates scoring to the HybridEngine and writes the result to cache.

    Args:
        user_id: Primary key of the user whose scores should be refreshed.
    """
    logger.info(
        "refresh_user_scores started",
        extra={"task_id": self.request.id, "user_id": user_id},
    )
    start = time.monotonic()

    try:
        engine = HybridEngine()
        recommendations = engine.recommend(user_id=user_id)
        set_cached_recommendations(user_id=user_id, ranked_list=recommendations)
    except Exception as exc:
        logger.exception(
            "refresh_user_scores failed",
            extra={
                "task_id": self.request.id,
                "user_id": user_id,
                "retries": self.request.retries,
                "exc": str(exc),
            },
        )
        raise

    duration = time.monotonic() - start
    logger.info(
        "refresh_user_scores completed",
        extra={
            "task_id": self.request.id,
            "user_id": user_id,
            "duration_seconds": round(duration, 3),
        },
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def train_collaborative_task(self) -> None:
    """
    Perform a full refit of the collaborative filter from interaction history.

    Triggered nightly via Celery beat or by the management command.
    Calls the collaborative engine's fit() method directly so the management
    command and the scheduled task share identical training logic.
    """
    logger.info(
        "train_collaborative_task started",
        extra={"task_id": self.request.id},
    )
    start = time.monotonic()

    try:
        # Import lazily to avoid circular-import issues at module load time.
        from recommendations.domain.engines.collaborative import CollaborativeEngine  # noqa: PLC0415

        engine = CollaborativeEngine()
        engine.fit()
    except Exception as exc:
        logger.exception(
            "train_collaborative_task failed",
            extra={
                "task_id": self.request.id,
                "retries": self.request.retries,
                "exc": str(exc),
            },
        )
        raise

    duration = time.monotonic() - start
    logger.info(
        "train_collaborative_task completed",
        extra={
            "task_id": self.request.id,
            "duration_seconds": round(duration, 3),
        },
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def train_content_based_task(self) -> None:
    """
    Rebuild the TF-IDF matrix from current Dataset metadata.

    Run after bulk dataset metadata updates to keep content-based
    recommendations aligned with the latest catalogue.
    """
    logger.info(
        "train_content_based_task started",
        extra={"task_id": self.request.id},
    )
    start = time.monotonic()

    try:
        # Import lazily to avoid circular-import issues at module load time.
        from recommendations.domain.engines.content_based import ContentBasedEngine  # noqa: PLC0415

        engine = ContentBasedEngine()
        engine.fit()
    except Exception as exc:
        logger.exception(
            "train_content_based_task failed",
            extra={
                "task_id": self.request.id,
                "retries": self.request.retries,
                "exc": str(exc),
            },
        )
        raise

    duration = time.monotonic() - start
    logger.info(
        "train_content_based_task completed",
        extra={
            "task_id": self.request.id,
            "duration_seconds": round(duration, 3),
        },
    )