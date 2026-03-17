# metadata/api/serializers.py
"""
Django REST Framework serializers for the Metadata Extraction API.

Translates between internal Python objects (Django models, dataclasses)
and JSON representations returned by the API. There is one serializer
per major API resource:

    PipelineRunSerializer        - read: monitor pipeline run state
    PipelineRunCreateSerializer  - write: trigger a new pipeline run
    MetadataResultSerializer     - read: retrieve extracted JSON schema output
    ColumnProfileSerializer      - read: expose per-column profile details

All serializers are read-only by default. Write serializers are
prefixed with 'Create' (e.g. PipelineRunCreateSerializer).

Import map (used by api/views.py):
    from .serializers import (
        PipelineRunSerializer,
        PipelineRunCreateSerializer,
        MetadataResultSerializer,
        ColumnProfileSerializer,
    )
"""

from __future__ import annotations

from rest_framework import serializers

from metadata.models import (
    ColumnProfile,
    MetadataResult,
    PipelineRun,
    RunStatus,
    SourceType,
)


# ---------------------------------------------------------------------------
# PipelineRunCreateSerializer  — POST /api/runs/
# ---------------------------------------------------------------------------

class PipelineRunCreateSerializer(serializers.Serializer):
    """
    Write-only serializer for triggering a new pipeline run.

    Validates the incoming POST body and exposes only the fields that the
    view needs to create a PipelineRun record and dispatch the Celery task.

    Fields:
        source              required  "csv" | "excel" | "sql"
        source_path         required  File path (csv/excel) or table name (sql).
        dataset_title       optional  Human-readable title for the JSON Schema.
        dataset_description optional  Description embedded in the JSON Schema.
        sql_schema          optional  DB schema/namespace (sql source only).
        sql_query           optional  Raw SELECT statement (sql source only).

    Validation rules:
        - source must be one of the SourceType choices.
        - source_path is always required (file path for csv/excel, table name
          for sql). It cannot be blank.
        - sql_schema and sql_query are only meaningful when source="sql";
          a warning is NOT raised for passing them with other sources — they
          are simply ignored downstream — but they are accepted here to avoid
          forcing callers to conditionally strip them.
    """

    source = serializers.ChoiceField(
        choices=SourceType.choices,
        help_text='Data source type. One of: "csv", "excel", "sql".',
    )
    source_path = serializers.CharField(
        max_length=1024,
        allow_blank=False,
        help_text="File path (csv/excel) or table name (sql).",
    )
    dataset_title = serializers.CharField(
        max_length=255,
        required=False,
        default="",
        allow_blank=True,
        help_text="Human-readable title forwarded to SchemaBuilder.",
    )
    dataset_description = serializers.CharField(
        required=False,
        default="",
        allow_blank=True,
        help_text="Optional description for the top-level JSON Schema document.",
    )

    # SQL-only extras --------------------------------------------------------
    sql_schema = serializers.CharField(
        required=False,
        default=None,
        allow_null=True,
        allow_blank=True,
        help_text='Database schema/namespace (e.g. "public"). SQL source only.',
    )
    sql_query = serializers.CharField(
        required=False,
        default=None,
        allow_null=True,
        allow_blank=True,
        help_text="Optional raw SELECT statement. SQL source only.",
    )

    def validate(self, attrs: dict) -> dict:
        """
        Cross-field validation.

        Ensures sql_schema and sql_query are only submitted alongside
        source="sql", and that source_path is non-empty.
        """
        source     = attrs.get("source", "")
        sql_schema = attrs.get("sql_schema")
        sql_query  = attrs.get("sql_query")

        if source != SourceType.SQL:
            if sql_schema:
                raise serializers.ValidationError(
                    {"sql_schema": "sql_schema is only valid when source='sql'."}
                )
            if sql_query:
                raise serializers.ValidationError(
                    {"sql_query": "sql_query is only valid when source='sql'."}
                )

        return attrs


# ---------------------------------------------------------------------------
# PipelineRunSerializer  — GET /api/runs/  and  GET /api/runs/<id>/
# ---------------------------------------------------------------------------

class PipelineRunSerializer(serializers.ModelSerializer):
    """
    Read-only serializer for PipelineRun.

    Exposes all fields needed to monitor a run, including timing data
    and any error message. Used for both the list and detail endpoints.

    The ``is_terminal`` computed field lets clients avoid parsing the
    status string to decide whether to stop polling.
    """

    is_terminal = serializers.BooleanField(
        read_only=True,
        help_text="True when status is SUCCESS or FAILED (final state).",
    )

    class Meta:
        model  = PipelineRun
        fields = [
            "id",
            "source",
            "source_path",
            "dataset_title",
            "dataset_description",
            "status",
            "is_terminal",
            "error_message",
            "elapsed_s",
            "stage_times",
            "created_at",
            "started_at",
            "finished_at",
        ]
        read_only_fields = fields


# ---------------------------------------------------------------------------
# MetadataResultSerializer  — GET /api/runs/<id>/schema/
# ---------------------------------------------------------------------------

class MetadataResultSerializer(serializers.ModelSerializer):
    """
    Read-only serializer for MetadataResult.

    Surfaces the full JSON Schema output (both as a string and as a parsed
    dict) alongside the diagnostic report produced by SchemaBuilder.

    ``run_id`` is included as a top-level field so the client can correlate
    the schema back to the originating run without parsing a nested object.
    """

    run_id = serializers.UUIDField(
        source="run.id",
        read_only=True,
        help_text="UUID of the parent PipelineRun.",
    )

    class Meta:
        model  = MetadataResult
        fields = [
            "run_id",
            "json_schema",
            "schema_dict",
            "schema_report",
            "column_count",
            "created_at",
        ]
        read_only_fields = fields


# ---------------------------------------------------------------------------
# ColumnProfileSerializer  — GET /api/runs/<id>/columns/
# ---------------------------------------------------------------------------

class ColumnProfileSerializer(serializers.ModelSerializer):
    """
    Read-only serializer for ColumnProfile.

    Exposes the most useful per-column attributes at the top level for
    easy filtering and display. The complete raw profile dict is also
    included under ``profile_data`` for clients that need every field
    produced by DataFrameProfiler and the enrichment stages.

    ``run_id`` is included for convenience when serializing profiles
    outside of the context of a single run (e.g. cross-run comparisons).
    """

    run_id = serializers.UUIDField(
        source="run.id",
        read_only=True,
        help_text="UUID of the parent PipelineRun.",
    )

    class Meta:
        model  = ColumnProfile
        fields = [
            "run_id",
            "column_name",
            "dtype",
            "semantic_type",
            "semantic_confidence",
            "nullable",
            "unique_count",
            "null_count",
            "profile_data",
        ]
        read_only_fields = fields