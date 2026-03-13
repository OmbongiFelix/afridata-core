"""
Django REST Framework serializers for the Metadata Extraction API.

Translates between internal Python objects (Django models, dataclasses)
and JSON representations returned by the API. There is one serializer
per major API resource:

    PipelineRunSerializer    - trigger and monitor pipeline runs
    MetadataResultSerializer - retrieve extracted JSON schema output
    ColumnProfileSerializer  - expose per-column profile details

All serializers are read-only by default. Write serializers are
prefixed with 'Create' (e.g. CreatePipelineRunSerializer).
"""
