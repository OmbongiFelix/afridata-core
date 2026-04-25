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

from .models import MetadataResult, PipelineRun


@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ("id", "source_type", "status", "created_by", "started_at")
    list_filter = ("status", "source_type")
    search_fields = ("created_by__username",)

    def get_readonly_fields(self, request, obj=None):
        """Mark all fields as readonly — admin is for monitoring, not editing."""
        if obj:
            return [field.name for field in obj._meta.get_fields()]
        return self.readonly_fields

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


class MetadataResultInline(admin.TabularInline):
    """Inline JSON schema preview for MetadataResult within PipelineRun."""

    model = MetadataResult
    extra = 0
    readonly_fields = ("schema_preview",)
    can_delete = False

    def schema_preview(self, obj):
        """Return a truncated preview of the JSON schema."""
        import json

        from django.utils.html import format_html

        try:
            pretty = json.dumps(obj.schema, indent=2)
            preview = pretty[:500] + ("..." if len(pretty) > 500 else "")
            return format_html("<pre style='white-space:pre-wrap'>{}</pre>", preview)
        except Exception:
            return str(obj.schema)

    schema_preview.short_description = "JSON Schema Preview"


@admin.register(MetadataResult)
class MetadataResultAdmin(admin.ModelAdmin):
    list_display = ("id", "pipeline_run", "created_at")
    list_filter = ("pipeline_run__status", "pipeline_run__source_type")
    search_fields = ("pipeline_run__created_by__username",)
    inlines = [MetadataResultInline]

    def get_readonly_fields(self, request, obj=None):
        """Mark all fields as readonly — admin is for monitoring, not editing."""
        if obj:
            return [field.name for field in obj._meta.get_fields()]
        return self.readonly_fields

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False