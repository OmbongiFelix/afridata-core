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

from .views import MetadataSchemaView, PipelineRunDetailView, PipelineRunListCreateView

app_name = "metadata"

urlpatterns = [
    path("runs/", PipelineRunListCreateView.as_view(), name="run-list-create"),
    path("runs/<int:pk>/", PipelineRunDetailView.as_view(), name="run-detail"),
    path("runs/<int:pk>/schema/", MetadataSchemaView.as_view(), name="run-schema"),
]