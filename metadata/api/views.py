"""
API views for the Metadata Extraction Pipeline.

Provides RESTful endpoints to trigger pipeline runs and retrieve
extracted metadata results. Uses DRF GenericAPIView and mixins.

Endpoints (registered in api/urls.py):
    POST /api/runs/          - trigger a new pipeline run
    GET  /api/runs/          - list all pipeline runs
    GET  /api/runs/<id>/     - retrieve a specific run
    GET  /api/runs/<id>/schema/ - retrieve the JSON schema output

Pipeline runs are dispatched asynchronously via Celery. The POST
endpoint returns immediately with a run_id for polling.
"""
