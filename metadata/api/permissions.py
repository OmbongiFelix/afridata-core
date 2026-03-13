"""
Custom permission classes for the Metadata Extraction API.

Extends DRF's BasePermission to enforce access control beyond
simple IsAuthenticated checks:

    IsPipelineAdmin  - can trigger runs and delete results
    IsResultViewer   - can only read completed schema outputs
    IsOwnerOrAdmin   - can only access their own pipeline runs

Assign via view's permission_classes attribute.
"""
