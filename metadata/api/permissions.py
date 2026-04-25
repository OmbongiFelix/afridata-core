"""
Custom permission classes for the Metadata Extraction API.

Extends DRF's BasePermission to enforce access control beyond
simple IsAuthenticated checks:

    IsPipelineAdmin  - can trigger runs and delete results
    IsResultViewer   - can only read completed schema outputs
    IsOwnerOrAdmin   - can only access their own pipeline runs

Assign via view's permission_classes attribute.
"""

from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsPipelineAdmin(BasePermission):
    """
    Grants full access to users who belong to the 'pipeline_admin' group.

    Permitted actions:
      - Trigger new pipeline runs (POST)
      - Delete pipeline results (DELETE)
      - All other read/write operations
    """

    message = "You must be a pipeline admin to perform this action."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.groups.filter(name="pipeline_admin").exists()
        )


class IsResultViewer(BasePermission):
    """
    Grants read-only access to completed pipeline schema outputs.

    Permitted actions:
      - GET, HEAD, OPTIONS (safe methods only)

    Denied actions:
      - Any write operation (POST, PUT, PATCH, DELETE)
    """

    message = "You have read-only access to pipeline results."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.method in SAFE_METHODS
        )


class IsOwnerOrAdmin(BasePermission):
    """
    Object-level permission that allows access only to the owner of a
    PipelineRun or to pipeline admins.

    Ownership is determined by comparing request.user against
    PipelineRun.created_by. Pipeline admins bypass the ownership check.
    """

    message = "You do not have permission to access this pipeline run."

    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        # Pipeline admins can access any object
        if request.user.groups.filter(name="pipeline_admin").exists():
            return True

        # All other authenticated users can only access their own runs
        return obj.created_by == request.user