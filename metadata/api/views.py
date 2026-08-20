# metadata/api/views.py
"""
API views for the Metadata Extraction Pipeline.

Provides RESTful endpoints to trigger pipeline runs and retrieve
extracted metadata results. Uses DRF GenericAPIView and mixins.

Endpoints (registered in api/urls.py):
    POST /api/runs/                  - trigger a new pipeline run
    GET  /api/runs/                  - list all pipeline runs
    GET  /api/runs/<id>/             - retrieve a specific run
    GET  /api/runs/<id>/schema/      - retrieve the JSON schema output
    GET  /api/runs/<id>/columns/     - list column profiles for a run

Pipeline runs are dispatched asynchronously via Celery. The POST
endpoint returns immediately with a run_id for polling.
"""

from __future__ import annotations

import logging
from typing import Type
from celery import Task

from rest_framework import generics, mixins, status
from rest_framework import request
from rest_framework.exceptions import NotFound, ValidationError as DRFValidationError
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import BasePermission, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from django.core.exceptions import ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework.serializers import BaseSerializer
from rest_framework.views import APIView
from rest_framework.serializers import Serializer

from .permissions import IsPipelineAdmin, IsOwnerOrAdmin, IsResultViewer

from metadata.models import ColumnProfile, MetadataResult, PipelineRun, RunStatus
from .serializers import (
    ColumnProfileSerializer,
    MetadataResultSerializer,
    PipelineRunCreateSerializer,
    PipelineRunSerializer,
)

from metadata.tasks import run_pipeline_task as _run_pipeline_task # Celery task
from django.db.models import QuerySet
from rest_framework.request import Request as DrfRequest


    

from typing import Any, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_run_or_404(pk: str) -> PipelineRun:
    """Return a PipelineRun by pk or raise DRF NotFound."""
    try:
        return PipelineRun.objects.get(pk=pk)
    except (PipelineRun.DoesNotExist, ValueError, DjangoValidationError):
        raise NotFound(detail=f"PipelineRun '{pk}' not found.")
    

# POST /api/runs/   — trigger a new pipeline run
# GET  /api/runs/   — list all pipeline runs
# ---------------------------------------------------------------------------

class PipelineRunListCreateView(
    mixins.ListModelMixin,
    mixins.CreateModelMixin,
    generics.GenericAPIView,
):
    """
    GET  — Return a paginated list of all PipelineRun records, newest first.
    POST — Validate the request body, create a PipelineRun in PENDING state,
           dispatch the Celery task, and return the run_id immediately.

    Request body (POST):
        source              str  required  "csv" | "excel" | "sql"
        source_path         str  required  file path or table name
        dataset_title       str  optional
        dataset_description str  optional

        # SQL-only extras (all optional):
        sql_schema          str  optional  database schema / namespace
        sql_query           str  optional  raw SELECT statement

    Response (POST, HTTP 202):
        {
            "id":          "<uuid>",
            "status":      "PENDING",
            "source":      "csv",
            "source_path": "data.csv",
            ...
        }
    """

    queryset         = PipelineRun.objects.all().order_by("-created_at")
    pagination_class = PageNumberPagination

    def get_permissions(self) -> list[BasePermission]:
        """
        POST  — pipeline admins only (can trigger runs).
        GET   — admins or read-only viewers (any authenticated user with a role).
        """
        if self.request.method == "POST":
            return [IsPipelineAdmin()]
        # For GET, accept either role; IsResultViewer already enforces SAFE_METHODS.
        return [IsResultViewer()]

    def get_serializer_class(self) -> Type[BaseSerializer]: # type: ignore[override]
        if self.request.method == "POST":
            return PipelineRunCreateSerializer
        return PipelineRunSerializer

    # --- GET ----------------------------------------------------------------

    def get(self, request: Request, *args, **kwargs) -> Response:
        """List all pipeline runs, ordered by creation time (newest first)."""
        return self.list(request, *args, **kwargs)

    # --- POST ---------------------------------------------------------------

    def post(self, request: Request, *args, **kwargs) -> Response:
        """
        Validate the payload, persist a PENDING run, enqueue the Celery task.

        Returns HTTP 202 Accepted so the caller knows work is in progress,
        not yet complete (which would be 201 Created).
        """
        serializer = PipelineRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict[str, Any], serializer.validated_data)

        # --- Create the run record -----------------------------------------
        run = PipelineRun.objects.create(
            source              = data["source"],
            source_path         = data.get("source_path", ""),
            dataset_title       = data.get("dataset_title", ""),
            dataset_description = data.get("dataset_description", ""),
            status              = RunStatus.PENDING,
            created_by          = request.user,  # required for IsOwnerOrAdmin checks
        )
        logger.info(
            "Created PipelineRun %s [source=%s, path=%s].",
            run.id, run.source, run.source_path,
        )
        
        run_pipeline_task: Task = _run_pipeline_task  # type: ignore[assignment]
        # --- Dispatch Celery task ------------------------------------------
        # In dev/test, CELERY_TASK_ALWAYS_EAGER=True runs the task inline.
        # In production, the task is sent to the broker asynchronously.
        # Either way, the run record is already in PENDING — the 202 response
        # is always returned so clients can poll for completion.
        try:
            run_pipeline_task.delay(
                run_id              = str(run.id),
                source              = run.source,
                source_path         = run.source_path,
                dataset_title       = run.dataset_title,
                dataset_description = run.dataset_description,
                sql_schema          = data.get("sql_schema"),
                sql_query           = data.get("sql_query"),
            )
            logger.info("Dispatched Celery task for PipelineRun %s.", run.id)
        except Exception:
            # Broker unavailable (dev without Redis) or eager-mode pipeline
            # failure. The run record already exists in PENDING state; log the
            # error and return the 202 so clients can still poll for status.
            logger.exception(
                "Failed to dispatch pipeline task for PipelineRun %s. "
                "Run remains in PENDING state.",
                run.id,
            )

        return Response(
            PipelineRunSerializer(run).data,
            status=status.HTTP_202_ACCEPTED,
        )


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/   — retrieve a specific run
# ---------------------------------------------------------------------------

class PipelineRunDetailView(
    mixins.RetrieveModelMixin,
    generics.GenericAPIView,
):
    """
    GET — Return the full detail of a single PipelineRun by its UUID.

    Useful for polling run status. The client should poll until
    ``status`` is ``SUCCESS`` or ``FAILED``.

    Response fields include:
        id, source, source_path, dataset_title, status,
        error_message, elapsed_s, stage_times,
        created_at, started_at, finished_at.
    """

    queryset            = PipelineRun.objects.all()
    serializer_class    = PipelineRunSerializer
    permission_classes  = [IsOwnerOrAdmin]

    def get(self, request: Request, pk: str, *args, **kwargs) -> Response:
        # get_object() calls check_object_permissions(), which triggers
        # IsOwnerOrAdmin.has_object_permission(). Using _get_run_or_404()
        # directly would bypass that check entirely.
        run = self.get_object()
        serializer = self.get_serializer(run)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/schema/   — retrieve the JSON schema output
# ---------------------------------------------------------------------------

class PipelineRunSchemaView(APIView):
    """
    GET — Return the MetadataResult (JSON Schema) for a completed run.

    Returns HTTP 404 if the run does not exist.
    Returns HTTP 409 if the run is not yet in SUCCESS state (schema not
    available until the pipeline completes successfully).

    Response body (HTTP 200):
        {
            "run_id":        "<uuid>",
            "json_schema":   "<JSON Schema draft-07 string>",
            "schema_dict":   { ... },
            "schema_report": { ... },
            "column_count":  42,
            "created_at":    "2024-01-01T00:00:00Z"
        }
    """

    permission_classes = [IsOwnerOrAdmin]

    def get(self, request: Request, pk: str, *args, **kwargs) -> Response:
        run = _get_run_or_404(pk)

        # APIView has no get_object(), so trigger the object-level permission
        # check manually — this is the standard DRF pattern for APIView subclasses.
        self.check_object_permissions(request, run)

        if run.status != RunStatus.SUCCESS:
            return Response(
                {
                    "detail": (
                        f"Schema is not available. "
                        f"Run status is '{run.status}'. "
                        f"Schema is only available for runs with status 'SUCCESS'."
                    ),
                    "run_id": str(run.id),
                    "status": run.status,
                },
                status=status.HTTP_409_CONFLICT,
            )

        try:
            result: MetadataResult = run.metadata_result # type: ignore[attr-defined]
        except MetadataResult.DoesNotExist:
            logger.error(
                "PipelineRun %s is SUCCESS but MetadataResult is missing.", run.id
            )
            raise NotFound(
                detail=(
                    "Run completed successfully but the schema record is missing. "
                    "This is unexpected — please contact support."
                )
            )

        serializer = MetadataResultSerializer(result)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/columns/   — list column profiles for a run
# ---------------------------------------------------------------------------

class PipelineRunColumnProfilesView(
    mixins.ListModelMixin,
    generics.GenericAPIView,
):
    """
    GET — Return all ColumnProfile records for a completed PipelineRun.

    Supports optional query param filtering:
        ?semantic_type=EMAIL     — filter by semantic type label
        ?nullable=true|false     — filter by nullability

    Returns HTTP 409 if the run is not yet in SUCCESS state.

    Response body (HTTP 200):
        [
            {
                "column_name":          "email",
                "dtype":                "object",
                "semantic_type":        "EMAIL",
                "semantic_confidence":  0.97,
                "nullable":             false,
                "unique_count":         1024,
                "null_count":           0,
                "profile_data":         { ... }
            },
            ...
        ]
    """

    serializer_class    = ColumnProfileSerializer
    permission_classes  = [IsOwnerOrAdmin]

    def get_queryset(self) -> QuerySet[ColumnProfile]:  # type: ignore[override]
        pk  = self.kwargs["pk"]
        run = _get_run_or_404(pk)

        # get_queryset() runs before DRF's automatic object-permission machinery,
        # so trigger the check explicitly here — same pattern as PipelineRunSchemaView.
        self.check_object_permissions(self.request, run)

        if run.status != RunStatus.SUCCESS:
            raise DRFValidationError(
                {
                    "detail": (
                        f"Column profiles are not available. "
                        f"Run status is '{run.status}'."
                    ),
                    "run_id": str(run.id),
                    "status": run.status,
                }
            )

        qs = ColumnProfile.objects.filter(run=run)

        
        
        

        # Optional filters
        request: DrfRequest = self.request  # type: ignore[assignment]
        semantic_type = request.query_params.get("semantic_type")
        if semantic_type:
            qs = qs.filter(semantic_type__iexact=semantic_type)

        nullable = request.query_params.get("nullable")
        if nullable is not None:
            qs = qs.filter(nullable=(nullable.lower() == "true"))

        return qs.order_by("column_name")

    def get(self, request: Request, pk: str, *args, **kwargs) -> Response:
        return self.list(request, *args, **kwargs)