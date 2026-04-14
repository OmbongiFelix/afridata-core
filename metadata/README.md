# Metadata Extraction Pipeline

A Django application (`metadata/`) that automatically ingests data from multiple sources, profiles columns, classifies semantic types, and produces enriched JSON Schema output via a REST API.

---

## Pipeline Overview

```
Adapters → DataFrame → Profiler → Extractor → Classifier → LLM → JSON Schema
Stage 1     Stage 2    Stage 3    Stage 3b     Stage 4    Stage 5   Stage 6
```

| Stage | Description |
|-------|-------------|
| 1 – Adapters | Ingest raw data from CSV, Excel, SQL, or S3 into a unified DataFrame |
| 2 – DataFrame | Normalised Pandas DataFrame passed downstream |
| 3 – Profiler | Compute per-column statistics (nulls, cardinality, sample values) |
| 3b – Extractor | Augment profiles with source-specific metadata (SQL constraints, Excel headers) |
| 4 – Classifier | Assign semantic types (`email`, `currency`, `date`, etc.) via an ML model |
| 5 – LLM Generator | Enrich columns with human-readable descriptions, tags, and business names |
| 6 – Schema Builder | Serialise enriched metadata to a valid JSON Schema (draft-07) |

---

## Project Structure

```
metadata/
├── adapters/
│   ├── base_connector.py       # Abstract base class for all adapters
│   ├── csv_connector.py        # CSV ingestion (local, glob, HTTP)
│   ├── excel_connector.py      # Excel .xlsx / .xls ingestion
│   ├── sql_connector.py        # SQL database ingestion via SQLAlchemy
│   └── s3_connector.py         # Optional AWS S3 ingestion
├── core/
│   ├── pipeline.py             # Main orchestrator – chains all stages
│   ├── profiler.py             # Column-level statistical profiling
│   ├── schema_builder.py       # JSON Schema output builder
│   ├── extractors/
│   │   ├── csv_excel_extractor.py  # Format-specific metadata augmentation
│   │   └── sql_extractor.py        # DB schema metadata (PKs, FKs, constraints)
│   └── enhancement/
│       ├── semantic_classifier.py  # ML-based semantic type classification
│       └── llm_generator.py        # LLM-powered metadata enrichment
├── api/
│   ├── serializers.py          # DRF serializers for pipeline models
│   ├── views.py                # DRF views – trigger runs, retrieve results
│   ├── urls.py                 # URL routing for all API endpoints
│   └── permissions.py          # Custom DRF permission classes
├── models.py                   # Django ORM models (PipelineRun, MetadataResult)
├── admin.py                    # Django admin registration
└── tests/
    └── tests.py                # Unit and integration test suite
```

---

## Quick Start

### 1. Trigger a pipeline run

```python
from metadata.core.pipeline import MetadataPipeline

result = MetadataPipeline(source='csv', path='data.csv').run()
print(result.schema)
```

### 2. Via the REST API

```
POST /api/metadata/runs/
```

Returns a `run_id` immediately. The pipeline executes asynchronously via Celery.

```
GET /api/metadata/runs/<id>/schema/
```

Retrieve the completed JSON Schema output.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/metadata/runs/` | Trigger a new pipeline run |
| `GET` | `/api/metadata/runs/` | List all pipeline runs |
| `GET` | `/api/metadata/runs/<id>/` | Retrieve a specific run |
| `GET` | `/api/metadata/runs/<id>/schema/` | Retrieve the JSON Schema output |

Include this app's URLs in the root `urls.py`:

```python
path('api/metadata/', include('metadata.api.urls')),
```

---

## Supported Data Sources

| Source | Connector | Notes |
|--------|-----------|-------|
| CSV | `CSVConnector` | Local paths, glob patterns, HTTP URLs |
| Excel | `ExcelConnector` | `.xlsx` (openpyxl) and `.xls` (xlrd) |
| SQL | `SQLConnector` | PostgreSQL, MySQL, SQLite via SQLAlchemy |
| AWS S3 | `S3Connector` | Optional – streams CSV/Excel from S3 without temp files |

---

## Configuration

### LLM Backend

Configure in Django settings:

```python
LLM_BACKEND = 'openai'   # or 'anthropic' | 'ollama'
LLM_MODEL   = 'gpt-4o-mini'
```

### SQL Credentials

Never hardcode credentials. Use environment variables or Django settings:

```
DB_PASSWORD=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
```

### ML Classifier Model

The pretrained scikit-learn model must be placed at:

```
metadata/ml_models/semantic_classifier.pkl
```

If the file is missing, the classifier falls back to regex-based heuristics and logs a warning.

---

## Permissions

| Permission Class | Access |
|-----------------|--------|
| `IsPipelineAdmin` | Trigger runs and delete results (requires `pipeline_admin` group) |
| `IsResultViewer` | Read-only access to completed runs |
| `IsOwnerOrAdmin` | Access only own pipeline runs |

---

## Running Tests

```bash
python manage.py test metadata
```

The test suite covers all pipeline stages. External calls (LLM, S3, SQL) are fully mocked — no live network access required.

| Test Class | Coverage |
|------------|----------|
| `TestCSVConnector` | Adapter ingestion and error handling |
| `TestSQLConnector` | Database extraction with SQLite fixture |
| `TestColumnProfiler` | Profile accuracy on known DataFrames |
| `TestSemanticClassifier` | Semantic type predictions |
| `TestLLMGenerator` | LLM call mocked with `unittest.mock` |
| `TestSchemaBuilder` | Valid JSON Schema output |
| `TestPipelineIntegration` | End-to-end run with CSV fixture |
| `TestPipelineAPI` | DRF test client for all endpoints |

Target: **80%+ coverage** on `core/` and `adapters/`.

---

## JSON Schema Output Format

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "<dataset name>",
  "type": "object",
  "properties": {
    "<column_name>": {
      "type": "string",
      "description": "LLM-generated plain-English description",
      "x-semantic-type": "email",
      "x-null-pct": 0.02
    }
  }
}
```

All output is validated against JSON Schema draft-07 before being returned. If validation fails for a column, it falls back to `type: "string"`.

---

## Key Design Decisions

- **`pipeline.py` is intentionally thin** — all business logic lives in stage modules. Keep it under 150 lines.
- **Adapters are pure ingestion** — no business logic; return a raw `pd.DataFrame` unchanged.
- **Profiling is synchronous and deterministic** — no external API calls before the ML classifier.
- **LLM calls are batched** — multiple columns are combined into a single prompt to minimise API usage.
- **Pipeline runs are async** — never call `pipeline.run()` synchronously inside a Django view; dispatch via Celery.
- **Credentials are never stored** — `source_config` stores env var names only, never actual secrets.