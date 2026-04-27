# metadata/api/tests/test_api.py
"""
Test suite for the Metadata Extraction API.

Covers:
    - PipelineRunListCreateView  (GET/POST /api/runs/)
    - PipelineRunDetailView      (GET /api/runs/<id>/)
    - PipelineRunSchemaView      (GET /api/runs/<id>/schema/)
    - PipelineRunColumnProfilesView (GET /api/runs/<id>/columns/)
    - Serializers: PipelineRunCreateSerializer, PipelineRunSerializer,
                   MetadataResultSerializer, ColumnProfileSerializer
    - Permissions: IsPipelineAdmin, IsResultViewer, IsOwnerOrAdmin

Run with:
    python manage.py test metadata
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APIRequestFactory, APITestCase

from metadata.models import (
    ColumnProfile,
    MetadataResult,
    PipelineRun,
    RunStatus,
    SourceType,
)
from metadata.api.serializers import (
    ColumnProfileSerializer,
    MetadataResultSerializer,
    PipelineRunCreateSerializer,
    PipelineRunSerializer,
)
from metadata.api.permissions import IsPipelineAdmin, IsResultViewer, IsOwnerOrAdmin
from metadata.api.views import _get_run_or_404


# ---------------------------------------------------------------------------
# Helpers / Base classes
# ---------------------------------------------------------------------------

def make_user(username: str, password: str = "pass") -> User:
    return User.objects.create_user(username=username, password=password)


def make_admin_user(username: str = "admin") -> User:
    user = make_user(username)
    group, _ = Group.objects.get_or_create(name="pipeline_admin")
    user.groups.add(group)
    return user


def make_pipeline_run(
    created_by: User | None = None,
    source: str = SourceType.CSV,
    source_path: str = "data.csv",
    status: str = RunStatus.PENDING,
    **kwargs,
) -> PipelineRun:
    run = PipelineRun.objects.create(
        source=source,
        source_path=source_path,
        status=status,
        **kwargs,
    )
    if created_by:
        run.created_by = created_by
        run.save()
    return run


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------

class IsPipelineAdminPermissionTest(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.permission = IsPipelineAdmin()
        self.view = MagicMock()

    def _request(self, user):
        request = self.factory.get("/")
        request.user = user
        return request

    def test_admin_user_is_allowed(self):
        user = make_admin_user()
        request = self._request(user)
        self.assertTrue(self.permission.has_permission(request, self.view))

    def test_non_admin_user_is_denied(self):
        user = make_user("regular")
        request = self._request(user)
        self.assertFalse(self.permission.has_permission(request, self.view))

    def test_unauthenticated_user_is_denied(self):
        from django.contrib.auth.models import AnonymousUser
        request = self._request(AnonymousUser())
        self.assertFalse(self.permission.has_permission(request, self.view))

    def test_message_is_set(self):
        self.assertIn("pipeline admin", self.permission.message.lower())


class IsResultViewerPermissionTest(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.permission = IsResultViewer()
        self.view = MagicMock()

    def _request(self, user, method="GET"):
        factory_method = getattr(self.factory, method.lower())
        request = factory_method("/")
        request.user = user
        return request

    def test_get_is_allowed_for_authenticated_user(self):
        user = make_user("viewer")
        request = self._request(user, "GET")
        self.assertTrue(self.permission.has_permission(request, self.view))

    def test_post_is_denied_for_authenticated_user(self):
        user = make_user("viewer")
        request = self._request(user, "POST")
        self.assertFalse(self.permission.has_permission(request, self.view))

    def test_delete_is_denied(self):
        user = make_user("viewer")
        request = self._request(user, "DELETE")
        self.assertFalse(self.permission.has_permission(request, self.view))

    def test_unauthenticated_user_is_denied(self):
        from django.contrib.auth.models import AnonymousUser
        request = self._request(AnonymousUser(), "GET")
        self.assertFalse(self.permission.has_permission(request, self.view))

    def test_head_is_allowed(self):
        user = make_user("viewer")
        request = self._request(user, "HEAD")
        self.assertTrue(self.permission.has_permission(request, self.view))


class IsOwnerOrAdminPermissionTest(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.permission = IsOwnerOrAdmin()
        self.view = MagicMock()
        self.owner = make_user("owner")
        self.other = make_user("other")
        self.admin = make_admin_user("admin_user")

    def _request(self, user):
        request = self.factory.get("/")
        request.user = user
        return request

    def test_has_permission_returns_true_for_authenticated(self):
        request = self._request(self.owner)
        self.assertTrue(self.permission.has_permission(request, self.view))

    def test_owner_can_access_own_object(self):
        run = make_pipeline_run(created_by=self.owner)
        request = self._request(self.owner)
        self.assertTrue(self.permission.has_object_permission(request, self.view, run))

    def test_non_owner_cannot_access_others_object(self):
        run = make_pipeline_run(created_by=self.owner)
        request = self._request(self.other)
        self.assertFalse(self.permission.has_object_permission(request, self.view, run))

    def test_admin_can_access_any_object(self):
        run = make_pipeline_run(created_by=self.owner)
        request = self._request(self.admin)
        self.assertTrue(self.permission.has_object_permission(request, self.view, run))


# ---------------------------------------------------------------------------
# Serializer tests
# ---------------------------------------------------------------------------

class PipelineRunCreateSerializerTest(TestCase):

    def _valid_csv_payload(self, **overrides):
        return {
            "source": SourceType.CSV,
            "source_path": "data/file.csv",
            **overrides,
        }

    def test_valid_csv_payload_is_accepted(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload())
        self.assertTrue(s.is_valid(), s.errors)

    def test_valid_excel_payload_is_accepted(self):
        s = PipelineRunCreateSerializer(data={
            "source": SourceType.EXCEL,
            "source_path": "data/file.xlsx",
        })
        self.assertTrue(s.is_valid(), s.errors)

    def test_valid_sql_payload_is_accepted(self):
        s = PipelineRunCreateSerializer(data={
            "source": SourceType.SQL,
            "source_path": "my_table",
            "sql_schema": "public",
            "sql_query": "SELECT * FROM my_table",
        })
        self.assertTrue(s.is_valid(), s.errors)

    def test_missing_source_is_invalid(self):
        s = PipelineRunCreateSerializer(data={"source_path": "data.csv"})
        self.assertFalse(s.is_valid())
        self.assertIn("source", s.errors)

    def test_missing_source_path_is_invalid(self):
        s = PipelineRunCreateSerializer(data={"source": SourceType.CSV})
        self.assertFalse(s.is_valid())
        self.assertIn("source_path", s.errors)

    def test_blank_source_path_is_invalid(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload(source_path=""))
        self.assertFalse(s.is_valid())
        self.assertIn("source_path", s.errors)

    def test_invalid_source_choice_is_rejected(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload(source="parquet"))
        self.assertFalse(s.is_valid())
        self.assertIn("source", s.errors)

    def test_sql_schema_with_non_sql_source_raises_error(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload(sql_schema="public"))
        self.assertFalse(s.is_valid())
        self.assertIn("sql_schema", s.errors)

    def test_sql_query_with_non_sql_source_raises_error(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload(sql_query="SELECT 1"))
        self.assertFalse(s.is_valid())
        self.assertIn("sql_query", s.errors)

    def test_optional_fields_default_correctly(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload())
        s.is_valid()
        self.assertEqual(s.validated_data["dataset_title"], "")
        self.assertEqual(s.validated_data["dataset_description"], "")
        self.assertIsNone(s.validated_data["sql_schema"])
        self.assertIsNone(s.validated_data["sql_query"])

    def test_dataset_title_and_description_are_accepted(self):
        s = PipelineRunCreateSerializer(data=self._valid_csv_payload(
            dataset_title="My Dataset",
            dataset_description="A test dataset.",
        ))
        self.assertTrue(s.is_valid(), s.errors)
        self.assertEqual(s.validated_data["dataset_title"], "My Dataset")


class PipelineRunSerializerTest(TestCase):

    def test_serializes_pipeline_run_fields(self):
        run = make_pipeline_run(status=RunStatus.PENDING)
        s = PipelineRunSerializer(run)
        data = s.data
        self.assertIn("id", data)
        self.assertIn("status", data)
        self.assertIn("source", data)
        self.assertIn("source_path", data)
        self.assertIn("is_terminal", data)
        self.assertIn("created_at", data)

    def test_is_terminal_true_for_success(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        s = PipelineRunSerializer(run)
        self.assertTrue(s.data["is_terminal"])

    def test_is_terminal_true_for_failed(self):
        run = make_pipeline_run(status=RunStatus.FAILED)
        s = PipelineRunSerializer(run)
        self.assertTrue(s.data["is_terminal"])

    def test_is_terminal_false_for_pending(self):
        run = make_pipeline_run(status=RunStatus.PENDING)
        s = PipelineRunSerializer(run)
        self.assertFalse(s.data["is_terminal"])

    def test_is_terminal_false_for_running(self):
        run = make_pipeline_run(status=RunStatus.RUNNING)
        s = PipelineRunSerializer(run)
        self.assertFalse(s.data["is_terminal"])


class MetadataResultSerializerTest(TestCase):

    def test_serializes_result_fields(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        result = MetadataResult.objects.create(
            run=run,
            json_schema='{"type": "object"}',
            schema_dict={"type": "object"},
            schema_report={},
            column_count=5,
        )
        s = MetadataResultSerializer(result)
        data = s.data
        self.assertEqual(str(data["run_id"]), str(run.id))
        self.assertIn("json_schema", data)
        self.assertIn("schema_dict", data)
        self.assertIn("column_count", data)
        self.assertEqual(data["column_count"], 5)


class ColumnProfileSerializerTest(TestCase):

    def test_serializes_column_profile_fields(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        profile = ColumnProfile.objects.create(
            run=run,
            column_name="email",
            dtype="object",
            semantic_type="EMAIL",
            semantic_confidence=0.97,
            nullable=False,
            unique_count=100,
            null_count=0,
            profile_data={"min": None, "max": None},
        )
        s = ColumnProfileSerializer(profile)
        data = s.data
        self.assertEqual(str(data["run_id"]), str(run.id))
        self.assertEqual(data["column_name"], "email")
        self.assertEqual(data["semantic_type"], "EMAIL")
        self.assertAlmostEqual(float(data["semantic_confidence"]), 0.97)
        self.assertFalse(data["nullable"])


# ---------------------------------------------------------------------------
# View tests — base setup
# ---------------------------------------------------------------------------

class BaseAPITestCase(APITestCase):
    """
    Provides authenticated client, admin client, and shared URL helpers.
    URL names assume urls.py registers views with the names below.
    Adjust if your url conf uses different names.
    """

    def setUp(self):
        self.client = APIClient()
        self.user = make_user("regular_user")
        self.admin = make_admin_user("admin_user")
        self.other_user = make_user("other_user")

    def auth_as(self, user: User) -> APIClient:
        client = APIClient()
        client.force_authenticate(user=user)
        return client

    # URL helpers — adjust names to match your urls.py
    def run_list_url(self):
        return reverse("pipeline-run-list-create")

    def run_detail_url(self, pk):
        return reverse("pipeline-run-detail", kwargs={"pk": str(pk)})

    def run_schema_url(self, pk):
        return reverse("pipeline-run-schema", kwargs={"pk": str(pk)})

    def run_columns_url(self, pk):
        return reverse("pipeline-run-columns", kwargs={"pk": str(pk)})


# ---------------------------------------------------------------------------
# GET /api/runs/  — list
# ---------------------------------------------------------------------------

class PipelineRunListViewTest(BaseAPITestCase):

    def test_unauthenticated_request_is_rejected(self):
        response = self.client.get(self.run_list_url())
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_authenticated_user_can_list_runs(self):
        make_pipeline_run()
        make_pipeline_run()
        client = self.auth_as(self.user)
        response = client.get(self.run_list_url())
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["results"]), 2)

    def test_list_returns_expected_fields(self):
        make_pipeline_run()
        client = self.auth_as(self.user)
        response = client.get(self.run_list_url())
        run = response.data["results"][0]
        for field in ("id", "status", "source", "source_path", "is_terminal"):
            self.assertIn(field, run)

    def test_list_orders_newest_first(self):
        run1 = make_pipeline_run(source_path="first.csv")
        run2 = make_pipeline_run(source_path="second.csv")
        client = self.auth_as(self.user)
        response = client.get(self.run_list_url())
        ids = [str(r["id"]) for r in response.data["results"]]
        self.assertGreater(ids.index(str(run1.id)), ids.index(str(run2.id)))


# ---------------------------------------------------------------------------
# POST /api/runs/  — create
# ---------------------------------------------------------------------------

class PipelineRunCreateViewTest(BaseAPITestCase):

    def _post(self, client, payload):
        return client.post(self.run_list_url(), data=payload, format="json")

    @patch("metadata.api.views._run_pipeline_task")
    def test_valid_csv_post_returns_202(self, mock_task):
        mock_task.delay = MagicMock()
        client = self.auth_as(self.user)
        response = self._post(client, {"source": SourceType.CSV, "source_path": "data.csv"})
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    @patch("metadata.api.views._run_pipeline_task")
    def test_run_created_with_pending_status(self, mock_task):
        mock_task.delay = MagicMock()
        client = self.auth_as(self.user)
        self._post(client, {"source": SourceType.CSV, "source_path": "data.csv"})
        run = PipelineRun.objects.order_by("-created_at").first()
        self.assertEqual(run.status, RunStatus.PENDING)

    @patch("metadata.api.views._run_pipeline_task")
    def test_celery_task_is_dispatched(self, mock_task):
        mock_task.delay = MagicMock()
        client = self.auth_as(self.user)
        self._post(client, {"source": SourceType.CSV, "source_path": "data.csv"})
        mock_task.delay.assert_called_once()

    @patch("metadata.api.views._run_pipeline_task")
    def test_response_contains_run_id_and_status(self, mock_task):
        mock_task.delay = MagicMock()
        client = self.auth_as(self.user)
        response = self._post(client, {"source": SourceType.CSV, "source_path": "data.csv"})
        self.assertIn("id", response.data)
        self.assertEqual(response.data["status"], RunStatus.PENDING)

    def test_missing_source_path_returns_400(self):
        client = self.auth_as(self.user)
        response = self._post(client, {"source": SourceType.CSV})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_source_type_returns_400(self):
        client = self.auth_as(self.user)
        response = self._post(client, {"source": "parquet", "source_path": "x.parquet"})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_sql_schema_with_csv_source_returns_400(self):
        client = self.auth_as(self.user)
        response = self._post(client, {
            "source": SourceType.CSV,
            "source_path": "data.csv",
            "sql_schema": "public",
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch("metadata.api.views._run_pipeline_task")
    def test_sql_post_dispatches_with_sql_extras(self, mock_task):
        mock_task.delay = MagicMock()
        client = self.auth_as(self.user)
        self._post(client, {
            "source": SourceType.SQL,
            "source_path": "orders",
            "sql_schema": "public",
            "sql_query": "SELECT * FROM orders",
        })
        call_kwargs = mock_task.delay.call_args.kwargs
        self.assertEqual(call_kwargs["sql_schema"], "public")
        self.assertEqual(call_kwargs["sql_query"], "SELECT * FROM orders")

    def test_unauthenticated_post_is_rejected(self):
        response = self._post(self.client, {"source": SourceType.CSV, "source_path": "x.csv"})
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/  — detail
# ---------------------------------------------------------------------------

class PipelineRunDetailViewTest(BaseAPITestCase):

    def test_retrieve_existing_run_returns_200(self):
        run = make_pipeline_run()
        client = self.auth_as(self.user)
        response = client.get(self.run_detail_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(str(response.data["id"]), str(run.id))

    def test_retrieve_nonexistent_run_returns_404(self):
        client = self.auth_as(self.user)
        response = client.get(self.run_detail_url(uuid.uuid4()))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_retrieve_invalid_pk_returns_404(self):
        client = self.auth_as(self.user)
        response = client.get(self.run_detail_url("not-a-uuid"))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_unauthenticated_request_returns_401(self):
        run = make_pipeline_run()
        response = self.client.get(self.run_detail_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_detail_includes_is_terminal_field(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        client = self.auth_as(self.user)
        response = client.get(self.run_detail_url(run.id))
        self.assertIn("is_terminal", response.data)
        self.assertTrue(response.data["is_terminal"])


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/schema/  — schema
# ---------------------------------------------------------------------------

class PipelineRunSchemaViewTest(BaseAPITestCase):

    def _make_result(self, run):
        return MetadataResult.objects.create(
            run=run,
            json_schema='{"type": "object"}',
            schema_dict={"type": "object"},
            schema_report={"warnings": []},
            column_count=3,
        )

    def test_success_run_returns_200_with_schema(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        self._make_result(run)
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("json_schema", response.data)
        self.assertEqual(str(response.data["run_id"]), str(run.id))

    def test_pending_run_returns_409(self):
        run = make_pipeline_run(status=RunStatus.PENDING)
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertIn("status", response.data)

    def test_running_run_returns_409(self):
        run = make_pipeline_run(status=RunStatus.RUNNING)
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)

    def test_failed_run_returns_409(self):
        run = make_pipeline_run(status=RunStatus.FAILED)
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)

    def test_nonexistent_run_returns_404(self):
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(uuid.uuid4()))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_success_run_missing_result_returns_404(self):
        """Run is SUCCESS but MetadataResult row was deleted / never created."""
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        # Do NOT create a MetadataResult
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_409_response_contains_run_id_and_status(self):
        run = make_pipeline_run(status=RunStatus.PENDING)
        client = self.auth_as(self.user)
        response = client.get(self.run_schema_url(run.id))
        self.assertEqual(str(response.data["run_id"]), str(run.id))
        self.assertEqual(response.data["status"], RunStatus.PENDING)

    def test_unauthenticated_request_returns_401(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        response = self.client.get(self.run_schema_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------------
# GET /api/runs/<id>/columns/  — column profiles
# ---------------------------------------------------------------------------

class PipelineRunColumnProfilesViewTest(BaseAPITestCase):

    def _make_profiles(self, run, count=3):
        profiles = []
        for i in range(count):
            profiles.append(ColumnProfile.objects.create(
                run=run,
                column_name=f"col_{i}",
                dtype="object",
                semantic_type="STRING",
                semantic_confidence=0.9,
                nullable=True,
                unique_count=10 * (i + 1),
                null_count=i,
                profile_data={},
            ))
        return profiles

    def test_success_run_returns_all_profiles(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        self._make_profiles(run, count=3)
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 3)

    def test_profiles_ordered_by_column_name(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        for name in ("zzz", "aaa", "mmm"):
            ColumnProfile.objects.create(
                run=run, column_name=name, dtype="object",
                semantic_type="STRING", semantic_confidence=0.9,
                nullable=True, unique_count=1, null_count=0, profile_data={},
            )
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id))
        names = [r["column_name"] for r in response.data["results"]]
        self.assertEqual(names, sorted(names))

    def test_pending_run_returns_400(self):
        run = make_pipeline_run(status=RunStatus.PENDING)
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_filter_by_semantic_type(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        ColumnProfile.objects.create(
            run=run, column_name="email_col", dtype="object",
            semantic_type="EMAIL", semantic_confidence=0.99,
            nullable=False, unique_count=50, null_count=0, profile_data={},
        )
        ColumnProfile.objects.create(
            run=run, column_name="name_col", dtype="object",
            semantic_type="STRING", semantic_confidence=0.8,
            nullable=True, unique_count=10, null_count=2, profile_data={},
        )
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id), {"semantic_type": "EMAIL"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["column_name"], "email_col")

    def test_filter_by_semantic_type_is_case_insensitive(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        ColumnProfile.objects.create(
            run=run, column_name="email_col", dtype="object",
            semantic_type="EMAIL", semantic_confidence=0.99,
            nullable=False, unique_count=50, null_count=0, profile_data={},
        )
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id), {"semantic_type": "email"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_nullable_true(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        ColumnProfile.objects.create(
            run=run, column_name="a", dtype="object", semantic_type="STRING",
            semantic_confidence=0.9, nullable=True, unique_count=1, null_count=1, profile_data={},
        )
        ColumnProfile.objects.create(
            run=run, column_name="b", dtype="object", semantic_type="STRING",
            semantic_confidence=0.9, nullable=False, unique_count=1, null_count=0, profile_data={},
        )
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id), {"nullable": "true"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["column_name"], "a")

    def test_filter_nullable_false(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        ColumnProfile.objects.create(
            run=run, column_name="a", dtype="object", semantic_type="STRING",
            semantic_confidence=0.9, nullable=True, unique_count=1, null_count=1, profile_data={},
        )
        ColumnProfile.objects.create(
            run=run, column_name="b", dtype="object", semantic_type="STRING",
            semantic_confidence=0.9, nullable=False, unique_count=1, null_count=0, profile_data={},
        )
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id), {"nullable": "false"})
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["column_name"], "b")

    def test_nonexistent_run_returns_404(self):
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(uuid.uuid4()))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_unauthenticated_request_returns_401(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        response = self.client.get(self.run_columns_url(run.id))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_profile_response_includes_run_id(self):
        run = make_pipeline_run(status=RunStatus.SUCCESS)
        self._make_profiles(run, count=1)
        client = self.auth_as(self.user)
        response = client.get(self.run_columns_url(run.id))
        profile = response.data["results"][0]
        self.assertEqual(str(profile["run_id"]), str(run.id))


# ---------------------------------------------------------------------------
# _get_run_or_404 helper
# ---------------------------------------------------------------------------

class GetRunOr404HelperTest(TestCase):

    def test_returns_run_for_valid_pk(self):
        run = make_pipeline_run()
        result = _get_run_or_404(str(run.id))
        self.assertEqual(result.id, run.id)

    def test_raises_not_found_for_unknown_uuid(self):
        from rest_framework.exceptions import NotFound
        with self.assertRaises(NotFound):
            _get_run_or_404(str(uuid.uuid4()))

    def test_raises_not_found_for_non_uuid_string(self):
        from rest_framework.exceptions import NotFound
        with self.assertRaises(NotFound):
            _get_run_or_404("not-a-uuid")