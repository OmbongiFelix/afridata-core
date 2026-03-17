# metadata/tasks.py
"""
Celery tasks for the Metadata Extraction Pipeline.

This module lives at  metadata/tasks.py  — one level above the api/
sub-package, directly inside the metadata Django app.

Rationale for placement
-----------------------
tasks.py belongs in  metadata/  (the app root), NOT in  metadata/api/ :

  metadata/
  ├── __init__.py
  ├── models.py          ← PipelineRun, MetadataResult, ColumnProfile
  ├── tasks.py           ← HERE  (Celery autodiscover scans app roots)
  └── api/
      ├── __init__.py
      ├── serializers.py
      ├── views.py
      └── urls.py

Celery's autodiscover_tasks() looks for a tasks module at the root of
each INSTALLED_APP.  Placing tasks.py inside metadata/api/ would require
manual task registration and would couple the async worker to the HTTP
layer, which defeats separation of concerns.  The api/ sub-package is
an HTTP transport layer; tasks.py is a domain/infrastructure concern
shared by both the API layer and any future management commands or
signals that need to trigger the pipeline.

Task
----
run_pipeline_task   Executes the full MetadataPipeline for a given run_id.
                    Called by  api/views.py  via  .delay()  immediately
                    after the PipelineRun record is created.

Error handling
--------------
Any unhandled exception is caught, the run is marked FAILED, and the
exception is re-raised so Celery records the failure in its result
backend and any configured monitoring (Flower, Sentry, etc.) is notified.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import shared_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_column_profiles_bulk(
    run,
    profiles: dict[str, dict],
) -> None:
    """
    Persist per-column profile data to the ColumnProfile table in one bulk
    INSERT, replacing any pre-existing rows for this run.

    Extracts the subset of fields that ColumnProfile stores as first-class
    columns; everything else is stored in profile_data as a catch-all blob.

    Args:
        run:      PipelineRun instance (must already be saved).
        profiles: Enriched profile dict from PipelineResult.profiles.
                  Key = column name, value = profile dict.
    """
    from metadata.models import ColumnProfile

    # Remove any stale profiles from a previous (e.g. retried) attempt.
    ColumnProfile.objects.filter(run=run).delete()

    column_profiles = []
    for col_name, profile in profiles.items():
        column_profiles.append(
            ColumnProfile(
                run                 = run,
                column_name         = col_name,
                dtype               = str(profile.get("dtype", "")),
                semantic_type       = profile.get("semantic_type", ""),
                semantic_confidence = profile.get("semantic_confidence"),
                nullable            = bool(profile.get("nullable", False)),
                unique_count        = profile.get("unique_count"),
                null_count          = profile.get("null_count"),
                profile_data        = profile,
            )
        )

    if column_profiles:
        ColumnProfile.objects.bulk_create(column_profiles)
        logger.debug(
            "Bulk-created %d ColumnProfile rows for run %s.",
            len(column_profiles),
            run.id,
        )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="metadata.tasks.run_pipeline_task",
    # Retry up to 2 times on transient errors (e.g. DB hiccup on startup),
    # with a 30-second back-off.  Validation / data errors are NOT retried
    # because the task catches them and marks the run FAILED explicitly.
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,          # acknowledge only after the task completes
    reject_on_worker_lost=True,
)
def run_pipeline_task(
    self,
    *,
    run_id:              str,
    source:              str,
    source_path:         str,
    dataset_title:       str       = "",
    dataset_description: str       = "",
    sql_schema:          str | None = None,
    sql_query:           str | None = None,
) -> dict[str, Any]:
    """
    Execute the full MetadataPipeline for a queued PipelineRun.

    This task is dispatched by  api/views.PipelineRunListCreateView.post()
    immediately after the PipelineRun record is created in PENDING state.

    Lifecycle
    ---------
    1. Fetch the PipelineRun record and call mark_running().
    2. Build and execute MetadataPipeline.run().
    3. Persist MetadataResult and ColumnProfile records from the result.
    4. Call mark_success() on the run.
    5. On any exception, call mark_failed() then re-raise.

    Args:
        run_id:              UUID string of the PipelineRun to execute.
        source:              "csv" | "excel" | "sql"
        source_path:         File path (csv/excel) or table name (sql).
        dataset_title:       Forwarded to MetadataPipeline.
        dataset_description: Forwarded to MetadataPipeline.
        sql_schema:          DB schema/namespace (sql source only).
        sql_query:           Raw SELECT statement (sql source only).

    Returns:
        A summary dict with run_id, status, elapsed_s, and column_count.
        Celery stores this in its result backend (if configured).

    Raises:
        Re-raises any exception after recording the failure on the run,
        so Celery marks the task as FAILURE in its result backend.
    """
    from metadata.models import MetadataResult, PipelineRun
    from core.pipeline import MetadataPipeline

    # ------------------------------------------------------------------
    # 1. Fetch the run record
    # ------------------------------------------------------------------
    try:
        run = PipelineRun.objects.get(pk=run_id)
    except PipelineRun.DoesNotExist:
        # The record was deleted between enqueue and execution — nothing
        # sensible to do other than log and bail out.
        logger.error(
            "run_pipeline_task: PipelineRun %s not found. Task abandoned.",
            run_id,
        )
        return {"run_id": run_id, "status": "NOT_FOUND"}

    logger.info(
        "run_pipeline_task: starting [run_id=%s, source=%s, path=%s].",
        run_id, source, source_path,
    )
    run.mark_running()

    # ------------------------------------------------------------------
    # 2. Build pipeline kwargs based on source type
    # ------------------------------------------------------------------
    pipeline_kwargs: dict[str, Any] = {
        "source":              source,
        "dataset_title":       dataset_title,
        "dataset_description": dataset_description,
    }

    if source in ("csv", "excel"):
        pipeline_kwargs["path"] = source_path

    elif source == "sql":
        # SQL sources need a live SQLAlchemy engine.  The engine is built
        # from settings so it does not travel over the message broker.
        try:
            from django.conf import settings
            from sqlalchemy import create_engine as _create_engine

            db_url = getattr(settings, "PIPELINE_SQL_DATABASE_URL", None)
            if not db_url:
                raise ValueError(
                    "settings.PIPELINE_SQL_DATABASE_URL is not configured. "
                    "It is required for source='sql' pipeline runs."
                )
            engine = _create_engine(db_url)
        except Exception as exc:
            logger.exception(
                "run_pipeline_task: failed to create SQLAlchemy engine for run %s.",
                run_id,
            )
            run.mark_failed(str(exc))
            raise

        pipeline_kwargs["engine"]     = engine
        pipeline_kwargs["table_name"] = source_path
        if sql_schema:
            pipeline_kwargs["schema"] = sql_schema
        if sql_query:
            pipeline_kwargs["sql_query"] = sql_query

    # ------------------------------------------------------------------
    # 3. Execute the pipeline
    # ------------------------------------------------------------------
    try:
        result = MetadataPipeline(**pipeline_kwargs).run()
    except Exception as exc:
        logger.exception(
            "run_pipeline_task: pipeline failed for run %s: %s",
            run_id, exc,
        )
        run.mark_failed(str(exc))
        raise  # let Celery record the task as FAILURE

    # ------------------------------------------------------------------
    # 4. Persist MetadataResult
    # ------------------------------------------------------------------
    try:
        MetadataResult.objects.update_or_create(
            run=run,
            defaults={
                "json_schema":   result.json_schema,
                "schema_dict":   result.schema,
                "schema_report": result.schema_report,
            },
        )
        logger.debug(
            "run_pipeline_task: MetadataResult saved for run %s.", run_id
        )
    except Exception as exc:
        logger.exception(
            "run_pipeline_task: failed to save MetadataResult for run %s: %s",
            run_id, exc,
        )
        run.mark_failed(f"Failed to persist schema result: {exc}")
        raise

    # ------------------------------------------------------------------
    # 5. Persist ColumnProfiles  (best-effort — does not fail the run)
    # ------------------------------------------------------------------
    try:
        _build_column_profiles_bulk(run, result.profiles)
    except Exception:
        # Non-fatal: the schema result is already saved.  Log the error
        # but let the run complete as SUCCESS.
        logger.exception(
            "run_pipeline_task: failed to save ColumnProfiles for run %s "
            "(non-fatal — schema result was persisted successfully).",
            run_id,
        )

    # ------------------------------------------------------------------
    # 6. Mark SUCCESS
    # ------------------------------------------------------------------
    run.mark_success(
        elapsed_s   = result.elapsed_s,
        stage_times = result.stage_times,
    )
    logger.info(
        "run_pipeline_task: completed [run_id=%s, elapsed=%.3fs].",
        run_id, result.elapsed_s,
    )

    column_count = len(result.schema.get("properties", {}))
    return {
        "run_id":       run_id,
        "status":       "SUCCESS",
        "elapsed_s":    result.elapsed_s,
        "column_count": column_count,
    }