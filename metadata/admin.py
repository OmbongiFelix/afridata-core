"""
Django admin registration for the Metadata Extraction Pipeline.

Provides a read-friendly admin interface for monitoring pipeline
runs without needing direct database access. Useful during
development and for ops teams.

Registers: PipelineRun, MetadataResult

PipelineRunAdmin features:
    - list_display: id, source_type, status, created_by, started_at
    - list_filter:  status, source_type
    - search_fields: created_by__username
    - readonly_fields: all (runs should not be edited via admin)
"""
from django.contrib import admin

# Register your models here.
 