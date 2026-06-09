# metadata/models.py
"""
Database models for the Metadata Extraction Pipeline.

Stores the state and results of every pipeline run so that outputs
are retrievable via the API without re-running the pipeline.

Models:
    PipelineRun     - one record per pipeline execution (status, timing)
    MetadataResult  - the JSON schema output attached to a PipelineRun
    ColumnProfile   - optional: persisted per-column profile records

PipelineRun.status values: PENDING | RUNNING | SUCCESS | FAILED
"""

import uuid

from django.db import models

from config import settings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RunStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    RUNNING = "RUNNING", "Running"
    SUCCESS = "SUCCESS", "Success"
    FAILED  = "FAILED",  "Failed"


class SourceType(models.TextChoices):
    CSV   = "csv",   "CSV"
    EXCEL = "excel", "Excel"
    SQL   = "sql",   "SQL"


# ---------------------------------------------------------------------------
# PipelineRun
# ---------------------------------------------------------------------------

class PipelineRun(models.Model):
    """
    One record per pipeline execution.

    Created with status=PENDING when a POST /api/runs/ request arrives.
    The Celery task transitions the record through RUNNING → SUCCESS/FAILED.

    Fields:
        id              UUID primary key — safe to expose in the API.
        source          Data source type: csv | excel | sql.
        source_path     File path or table name used as input.
        dataset_title   Human-readable title forwarded to SchemaBuilder.
        dataset_description
                        Optional description forwarded to SchemaBuilder.
        status          Current execution state (see RunStatus).
        error_message   Populated when status=FAILED; empty otherwise.
        elapsed_s       Wall-clock seconds for the complete pipeline run.
                        NULL while the run is still in progress.
        stage_times     JSON dict of per-stage timings from PipelineResult.
                        NULL while the run is still in progress.
        created_at      When the run record was first inserted.
        started_at      When the Celery task began executing.
        finished_at     When the Celery task completed (success or failure).
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    # --- Source config -----------------------------------------------------
    source = models.CharField(
        max_length=10,
        choices=SourceType.choices,
    )
    source_path = models.TextField(
        blank=True,
        default="",
        help_text="File path (csv/excel) or table name (sql).",
    )
    dataset_title = models.CharField(max_length=255, blank=True, default="")
    dataset_description = models.TextField(blank=True, default="")

    # --- Execution state ---------------------------------------------------
    status = models.CharField(
        max_length=10,
        choices=RunStatus.choices,
        default=RunStatus.PENDING,
        db_index=True,
    )
    error_message = models.TextField(blank=True, default="")

    # --- Timing ------------------------------------------------------------
    elapsed_s = models.FloatField(
        null=True,
        blank=True,
        help_text="Total wall-clock seconds for the complete run.",
    )
    stage_times = models.JSONField(
        null=True,
        blank=True,
        help_text="Per-stage timing dict keyed by stage name.",
    )
    
    # 
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="pipeline_runs",
    )

    # --- Timestamps --------------------------------------------------------
    created_at  = models.DateTimeField(auto_now_add=True)
    started_at  = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Pipeline Run"
        verbose_name_plural = "Pipeline Runs"

    def __str__(self) -> str:
        return f"PipelineRun({self.id}, source={self.source}, status={self.status})"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True if the run has reached a final state (success or failed)."""
        return self.status in (RunStatus.SUCCESS, RunStatus.FAILED)

    def mark_running(self) -> None:
        """Transition status → RUNNING and record start timestamp."""
        from django.utils import timezone
        self.status     = RunStatus.RUNNING
        self.started_at = timezone.now()
        self.save(update_fields=["status", "started_at"])

    def mark_success(self, elapsed_s: float, stage_times: dict) -> None:
        """Transition status → SUCCESS and persist timing data."""
        from django.utils import timezone
        self.status      = RunStatus.SUCCESS
        self.elapsed_s   = elapsed_s
        self.stage_times = stage_times
        self.finished_at = timezone.now()
        self.save(update_fields=["status", "elapsed_s", "stage_times", "finished_at"])

    def mark_failed(self, error_message: str) -> None:
        """Transition status → FAILED and record the error message."""
        from django.utils import timezone
        self.status        = RunStatus.FAILED
        self.error_message = error_message
        self.finished_at   = timezone.now()
        self.save(update_fields=["status", "error_message", "finished_at"])


# ---------------------------------------------------------------------------
# MetadataResult
# ---------------------------------------------------------------------------

class MetadataResult(models.Model):
    """
    JSON Schema output produced by a successful PipelineRun.

    Holds the three top-level outputs from PipelineResult:
        - json_schema   — pretty-printed JSON Schema draft-07 string
        - schema_dict   — raw schema dict for server-side inspection
        - schema_report — diagnostic summary dict from SchemaBuilder

    There is a strict 1-to-1 relationship with PipelineRun; only one
    MetadataResult may exist per run.

    Fields:
        run             OneToOne FK to PipelineRun.
        json_schema     Full JSON Schema draft-07 string.
        schema_dict     Raw schema stored as a JSON field for querying.
        schema_report   Diagnostic report dict (type distributions, etc.).
        column_count    Number of properties in the schema (for quick reads).
        created_at      Timestamp of record creation.
    """

    run = models.OneToOneField(
        PipelineRun,
        on_delete=models.CASCADE,
        related_name="metadata_result",
    )
    run_id: uuid.UUID  # tells the type checker this attribute exists

    json_schema   = models.TextField(help_text="Pretty-printed JSON Schema draft-07.")
    schema_dict   = models.JSONField(help_text="Raw JSON Schema dict.")
    schema_report = models.JSONField(
        default=dict,
        help_text="Diagnostic summary from SchemaBuilder.schema_report().",
    )

    column_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of columns (properties) in the schema.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Metadata Result"
        verbose_name_plural = "Metadata Results"

    def __str__(self) -> str:
        return f"MetadataResult(run={self.run_id}, columns={self.column_count})"

    def save(self, *args, **kwargs):
        # Auto-populate column_count from schema_dict if not explicitly set.
        if self.schema_dict and not self.column_count:
            self.column_count = len(self.schema_dict.get("properties", {}))
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------

class ColumnProfile(models.Model):
    """
    Optional per-column profile record persisted from PipelineResult.profiles.

    Storing profiles individually allows fine-grained API queries such as
    "list all columns with semantic_type=EMAIL" across multiple runs.

    Fields:
        run             FK to the parent PipelineRun.
        column_name     Column name as it appears in the source dataset.
        dtype           Pandas dtype string (e.g. "object", "float64").
        semantic_type   Semantic type label from SemanticClassifier.
        semantic_confidence
                        Confidence score [0.0, 1.0] from SemanticClassifier.
        nullable        Whether the column contains any null values.
        unique_count    Approximate distinct value count.
        null_count      Number of null / NaN values in the column.
        profile_data    Full profile dict for this column (catch-all storage).
    """

    run = models.ForeignKey(
        PipelineRun,
        on_delete=models.CASCADE,
        related_name="column_profiles",
    )
    run_id: uuid.UUID  # tells the type checker this attribute exists

    column_name = models.CharField(max_length=255)
    dtype       = models.CharField(max_length=64, blank=True, default="")

    # Semantic classification
    semantic_type       = models.CharField(max_length=128, blank=True, default="")
    semantic_confidence = models.FloatField(null=True, blank=True)

    # Quick-access statistics
    nullable     = models.BooleanField(default=False)
    unique_count = models.PositiveIntegerField(null=True, blank=True)
    null_count   = models.PositiveIntegerField(null=True, blank=True)

    # Full profile blob
    profile_data = models.JSONField(
        default=dict,
        help_text="Complete profile dict for this column from DataFrameProfiler.",
    )

    class Meta:
        ordering = ["run", "column_name"]
        unique_together = [("run", "column_name")]
        verbose_name = "Column Profile"
        verbose_name_plural = "Column Profiles"
        indexes = [
            models.Index(fields=["semantic_type"]),
            models.Index(fields=["run", "semantic_type"]),
        ]

    def __str__(self) -> str:
        return (
            f"ColumnProfile(run={self.run_id}, "
            f"column={self.column_name!r}, "
            f"type={self.semantic_type})"
        )