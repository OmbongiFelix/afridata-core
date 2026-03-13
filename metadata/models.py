"""
Database models for the Metadata Extraction Pipeline.

Stores the state and results of every pipeline run so that outputs
are retrievable via the API without re-running the pipeline.

Models:
    PipelineRun     - one record per pipeline execution (status, timing)
    MetadataResult  - the JSON schema output attached to a PipelineRun
    ColumnProfile   - optional: persisted per-column profile records

PipelineRun.status values: PENDING | RUNNING | SUCCESS | FAILED
"""


from django.db import models

# Create your models here.
