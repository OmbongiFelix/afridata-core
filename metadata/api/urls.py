#metadata/api/urls.py
"""
URL configuration for the Metadata Extraction API.

Registers all API routes using DRF's DefaultRouter and manual
path() entries. Include this file in the project's root urls.py:

    path('api/metadata/', include('metadata.api.urls')),

Route map:
    runs/                  → PipelineRunListCreateView
    runs/<int:pk>/         → PipelineRunDetailView
    runs/<int:pk>/schema/  → MetadataSchemaView
"""

from django.urls import path

from .views import (
    PipelineRunSchemaView as MetadataSchemaView, 
    PipelineRunDetailView, 
    PipelineRunListCreateView,
    PipelineRunColumnProfilesView,
)

app_name = "metadata"


urlpatterns = [
    path(
        "runs/",
        PipelineRunListCreateView.as_view(),
        name="pipeline-run-list-create",
    ),
    path(
        "runs/<uuid:pk>/",
        PipelineRunDetailView.as_view(),
        name="pipeline-run-detail",
    ),
    path(
        "runs/<uuid:pk>/schema/",
        MetadataSchemaView.as_view(),
        name="pipeline-run-schema",
    ),
    path(
        "runs/<uuid:pk>/columns/",
        PipelineRunColumnProfilesView.as_view(),
        name="pipeline-run-columns",
    ),
]

