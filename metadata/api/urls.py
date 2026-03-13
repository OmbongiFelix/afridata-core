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
