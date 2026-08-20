# AfriData Core

**AfriData Core** is a Django-based REST API platform for ingesting structured datasets, automatically inferring rich metadata through a semantic classifier pipeline, and serving personalised dataset recommendations via a weighted hybrid of Collaborative Filtering and Content-Based Filtering.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Apps](#apps)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [Seeding Demo Data](#seeding-demo-data)
- [Training the Recommendation Engines](#training-the-recommendation-engines)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Key Design Decisions](#key-design-decisions)

---

## Architecture Overview

```
                       ┌──────────────────────────────────────┐
                       │           REST API Layer              │
                       │  POST /api/metadata/runs/             │
                       │  GET  /api/metadata/runs/<id>/        │
                       │  GET  /api/recommendations/           │
                       └───────────────┬──────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────┐
              │               metadata app                     │
              │  Adapters → Profiler → Classifier → LLM →     │
              │  Schema Builder  (async via Celery)            │
              └────────────────────────┬───────────────────────┘
                                       │  syncs DatasetProxy
              ┌────────────────────────▼───────────────────────┐
              │           recommendations app                  │
              │  Candidate Gen → CF Engine + CBF Engine →     │
              │  Hybrid Fusion → Ranking → Redis Cache        │
              └────────────────────────────────────────────────┘
```

The platform is split into two Django apps:

| App | Role |
|-----|------|
| **`metadata`** | Ingests CSV, Excel, and SQL datasets; profiles columns; classifies semantic types; enriches metadata with an LLM; produces JSON Schema output. |
| **`recommendations`** | Delivers personalised Top-N dataset recommendations using a weighted hybrid engine (Collaborative Filtering + Content-Based Filtering), served via REST with Redis caching. |

---

## Apps

### `metadata` — Semantic Metadata Pipeline

```
Adapters → DataFrame → Profiler → Semantic Classifier → LLM Enrichment → JSON Schema
```

| Stage | Description |
|-------|-------------|
| **Adapters** | Ingest CSV, Excel (`.xlsx`/`.xls`), or SQL databases into a unified Pandas DataFrame |
| **Profiler** | Compute per-column statistics: null rates, cardinality, sample values, data types |
| **Semantic Classifier** | ML-based type inference: `email`, `currency`, `date`, `lat/lon`, `ID`, etc. |
| **LLM Enrichment** | Generate human-readable descriptions and business names via Gemini, OpenAI, or Anthropic |
| **Schema Builder** | Serialise to JSON Schema draft-07 with custom `x-semantic-type` and `x-null-pct` extensions |

Runs are **async by default** — a `POST` returns a `PipelineRun` UUID immediately; the pipeline executes via Celery and results are polled via `GET /api/metadata/runs/<id>/schema/`.

---

### `recommendations` — Hybrid Recommendation Engine

```
User Data → Candidate Gen → S_CF + S_CBF → Hybrid Fusion → Ranking → Top-N → Cache → API
```

**Fusion formula:** `S_hybrid = α · S_CF + (1 − α) · S_CBF`

| Stage | Description |
|-------|-------------|
| **Candidate Generation** | Filter seen items; cap pool size; return `CandidateSet` |
| **Collaborative (S_CF)** | Matrix Factorisation (ALS or truncated SVD) on `UserInteraction` history |
| **Content-Based (S_CBF)** | TF-IDF cosine similarity between user profile vector and dataset metadata |
| **Hybrid Fusion** | Weighted blend of both scores; auto-fallback to CBF-only for cold-start users |
| **Ranking** | Sort by `S_hybrid`; optional MMR diversity re-ranking; Top-N cutoff |
| **Cache** | Redis — key: `rec:user:{user_id}`, TTL: 1 hour; invalidated automatically on interaction save/delete |

---

## Requirements

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (package manager)
- SQLite (default) or PostgreSQL for production
- Redis (optional for local dev — Celery falls back to synchronous mode)

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/afridata-core
cd afridata-core
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env   # or edit .env directly
```

See [Environment Variables](#environment-variables) for all options.

### 3. Run migrations

```bash
uv run python manage.py migrate
```

### 4. Seed demo data (creates admin user + API token)

```bash
uv run python manage.py seed_demo
```

This outputs your **API token** — keep it handy for the next steps.

### 5. Start the development server

```bash
uv run python manage.py runserver
```

### 6. Ingest a dataset

```powershell
$TOKEN = "<your-api-token>"

Invoke-RestMethod -Uri "http://localhost:8000/api/metadata/runs/" -Method POST `
  -Headers @{Authorization="Token $TOKEN"; "Content-Type"="application/json"} `
  -Body '{"source": "csv", "source_path": "metadata/tests/fixtures/sample.csv", "dataset_title": "My Dataset"}'
```

```bash
# Linux / macOS
curl -X POST http://localhost:8000/api/metadata/runs/ \
  -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source": "csv", "source_path": "metadata/tests/fixtures/sample.csv", "dataset_title": "My Dataset"}'
```

### 7. Get recommendations

```bash
# Content-based
curl http://localhost:8000/api/recommendations/?strategy=content \
  -H "Authorization: Token $TOKEN"

# Collaborative
curl http://localhost:8000/api/recommendations/?strategy=collaborative \
  -H "Authorization: Token $TOKEN"

# Hybrid (default)
curl http://localhost:8000/api/recommendations/?strategy=hybrid \
  -H "Authorization: Token $TOKEN"
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Django
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost, 127.0.0.1

# Database (leave empty for SQLite default)
DB_CONNECTION_STRING=

# LLM Backend — selects which provider to use for metadata enrichment
LLM_BACKEND=gemini          # or: openai | anthropic
LLM_MODEL=gemini-1.5-pro    # e.g. gpt-4o-mini, claude-3-haiku-20240307

# LLM API Keys (only the one matching LLM_BACKEND is required)
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Redis (optional — used for Celery broker and recommendations cache)
REDIS_URL=redis://127.0.0.1:6379

# Celery
# True  → tasks run inline in the same process (local dev, no Redis required)
# False → tasks are sent to the broker asynchronously (production)
CELERY_TASK_ALWAYS_EAGER=True
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

## API Reference

### Authentication

All endpoints require a **Token** in the `Authorization` header:

```
Authorization: Token <your-api-token>
```

Obtain a token:

```bash
# Get a token via the auth endpoint
curl -X POST http://localhost:8000/api/auth/token/ \
  -d "username=admin&password=afridata2024"
```

---

### Metadata Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/metadata/runs/` | `pipeline_admin` group | Trigger a new metadata pipeline run |
| `GET` | `/api/metadata/runs/` | Authenticated | List all pipeline runs (paginated) |
| `GET` | `/api/metadata/runs/<uuid>/` | Owner or Admin | Poll run status |
| `GET` | `/api/metadata/runs/<uuid>/schema/` | Owner or Admin | Retrieve the inferred JSON Schema (only when `status=SUCCESS`) |
| `GET` | `/api/metadata/runs/<uuid>/columns/` | Owner or Admin | Retrieve per-column profiles |

#### POST `/api/metadata/runs/` — Request Body

```json
{
  "source": "csv",
  "source_path": "path/to/file.csv",
  "dataset_title": "My Dataset",
  "dataset_description": "Optional description"
}
```

| Field | Values | Required |
|-------|--------|----------|
| `source` | `csv`, `excel`, `sql` | ✅ |
| `source_path` | File path or SQL connection string | ✅ |
| `dataset_title` | String | ✅ |
| `dataset_description` | String | |
| `sql_schema` | SQL schema name (SQL source only) | |
| `sql_query` | Custom SQL query (SQL source only) | |

#### Run Status Values

| Status | Meaning |
|--------|---------|
| `PENDING` | Accepted; queued for processing |
| `RUNNING` | Pipeline is actively executing |
| `SUCCESS` | Completed — schema available |
| `FAILED` | Pipeline error — see `error_message` |

---

### Recommendations Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/recommendations/` | Authenticated | Get Top-N personalised recommendations |
| `POST` | `/api/recommendations/feedback/` | Authenticated | Submit explicit feedback/rating |

#### GET `/api/recommendations/` — Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `hybrid` | `hybrid`, `content`, or `collaborative` |
| `top_n` | `10` | Number of recommendations to return |
| `alpha` | `0.5` | Blend weight: `1.0` = CF only, `0.0` = CBF only |

#### GET Response

```json
{
  "recommendations": [
    {
      "dataset_id": "1001",
      "title": "Kenya Health Indicators 2023",
      "rank": 1,
      "s_hybrid": 0.87,
      "confidence": "high"
    }
  ],
  "alpha": 0.5,
  "top_n": 10,
  "generated_at": "2026-08-20T06:42:58Z"
}
```

#### POST `/api/recommendations/feedback/` — Request Body

```json
{
  "dataset_id": "1001",
  "interaction_type": "view",
  "rating": 4.5
}
```

| `interaction_type` | Description |
|--------------------|-------------|
| `view` | User viewed the dataset detail |
| `download` | User downloaded the dataset |
| `rating` | Explicit numeric rating (requires `rating` field) |

---

## Authentication

### Permission Classes

| Class | Who can use it |
|-------|----------------|
| `IsPipelineAdmin` | Users in the `pipeline_admin` group — can trigger runs and delete results |
| `IsResultViewer` | Any authenticated user — read-only access to completed runs |
| `IsOwnerOrAdmin` | Authenticated users can access only their own runs; admins see all |

Unauthenticated requests return **401 Unauthorized** with a `WWW-Authenticate: Token` header.

---

## Seeding Demo Data

The `seed_demo` management command sets up a complete local environment in one step:

```bash
uv run python manage.py seed_demo
# — or —
uv run python manage.py seed_demo --username alice --password secret123
```

It creates:
- `pipeline_admin` group
- An admin user and adds them to the group
- A DRF API Token for the user
- 5 sample `DatasetProxy` records (Kenya, Nigeria, Ghana, South Africa, Ethiopia datasets)

Output includes the token and ready-to-paste curl/PowerShell commands.

To regenerate the token:

```bash
uv run python manage.py seed_demo --reset-token
```

---

## Training the Recommendation Engines

The engines serve **0.0 scores** until trained. Run the following after collecting `UserInteraction` data:

```bash
# 1. Fit the collaborative filter (Matrix Factorisation)
uv run python manage.py train_collaborative --factors 50 --epochs 20 --evaluate

# 2. Build the TF-IDF content matrix from DatasetProxy metadata
uv run python manage.py train_content_based --max-features 10000

# 3. Propagate new models to all user caches
uv run python manage.py rebuild_index
```

Repeat this sequence whenever interaction data or dataset metadata changes significantly.

---

## Running Tests

```bash
# API + recommendations tests (fastest — ~5 min)
uv run python manage.py test metadata.tests.test_api recommendations.tests --verbosity=2

# Full test suite
uv run python manage.py test metadata recommendations --verbosity=1
```

**Current result: 119 tests — OK (0 failures, 0 errors)**

| Test Module | Scope |
|-------------|-------|
| `metadata.tests.test_api` | DRF views, serializers, permissions, auth |
| `metadata.tests.test_connectors` | CSV, Excel, SQL adapter ingestion |
| `metadata.tests.test_profiling` | Column profiler accuracy |
| `metadata.tests.test_generation` | LLM enrichment (mocked) |
| `metadata.tests.test_integration` | Full pipeline end-to-end |
| `recommendations.tests.tests` | Integration: pipeline + API + cache |
| `recommendations.tests.test_hybrid_engine` | Fusion formula, alpha edge cases |
| `recommendations.tests.test_ranking` | Score ordering, Top-N, MMR diversity |
| `recommendations.tests.test_candidate_generation` | Seen-item filtering, cold-start |

No external services are required to run the tests — all LLM, Redis, S3, Celery, and SQL calls are mocked.

---

## Project Structure

```
afridata-core/
├── config/
│   ├── settings.py             # Django settings (auth, cache, Celery, LLM)
│   ├── urls.py                 # Root URL routing
│   ├── wsgi.py / asgi.py       # WSGI/ASGI entry points
│   └── celery.py               # Celery app configuration
│
├── metadata/                   # Metadata extraction pipeline app
│   ├── adapters/               # Source connectors: CSV, Excel, SQL
│   ├── core/
│   │   ├── pipeline.py         # Orchestrator — chains all pipeline stages
│   │   ├── profiler.py         # Column-level statistical profiling
│   │   ├── schema_builder.py   # JSON Schema draft-07 output builder
│   │   ├── extractors/         # Source-specific metadata augmentation
│   │   └── enhancement/
│   │       ├── semantic_classifier.py  # ML-based semantic type inference
│   │       └── llm_generator.py        # LLM metadata enrichment
│   ├── api/
│   │   ├── views.py            # DRF views: trigger runs, poll status, retrieve schema
│   │   ├── serializers.py      # DRF serializers for PipelineRun, MetadataResult
│   │   ├── urls.py             # URL routing for all metadata endpoints
│   │   └── permissions.py      # IsPipelineAdmin, IsResultViewer, IsOwnerOrAdmin
│   ├── management/commands/
│   │   └── seed_demo.py        # Seed demo user, token, and DatasetProxy records
│   ├── migrations/             # Django DB migrations
│   ├── models.py               # PipelineRun, MetadataResult ORM models
│   └── tests/
│       ├── fixtures/           # sample.csv, sample.xlsx, sample.db
│       └── test_*.py           # Unit and integration tests
│
├── recommendations/            # Hybrid recommendation engine app
│   ├── domain/
│   │   ├── schemas.py          # CandidateSet, ScoredCandidate, RankedList, EngineConfig
│   │   ├── ranking.py          # Sort, Top-N cutoff, MMR diversity re-ranking
│   │   ├── evaluation.py       # Offline metrics: Precision@K, Recall@K, NDCG@K
│   │   └── engines/
│   │       ├── candidate_generation.py # Stage 2: build eligible item pool
│   │       ├── collaborative.py        # Stage 3a: ALS/SVD scoring → S_CF
│   │       ├── content_based.py        # Stage 3b: TF-IDF cosine → S_CBF
│   │       └── hybrid.py               # Stage 4: fuse, normalise, rank
│   ├── infrastructure/
│   │   ├── persistence.py      # All ORM queries — only layer that touches models.py
│   │   ├── model_store.py      # Collaborative model weights (local / S3)
│   │   ├── vector_store.py     # TF-IDF sparse matrix (scipy / S3)
│   │   └── cache.py            # Redis get/set/invalidate per-user cache
│   ├── api/
│   │   ├── views.py            # GET recommendations, POST feedback
│   │   ├── serializers.py      # RecommendationListSerializer, FeedbackSerializer
│   │   └── urls.py             # URL routing for recommendations endpoints
│   ├── management/commands/
│   │   ├── train_collaborative.py   # Fit CF model from interaction history
│   │   ├── train_content_based.py   # Build TF-IDF matrix from dataset metadata
│   │   └── rebuild_index.py         # Invalidate caches; recompute all user Top-N
│   ├── signals.py              # post_save/post_delete → Celery cache invalidation
│   ├── tasks.py                # Celery tasks: refresh_user_scores, train_*
│   ├── models.py               # UserInteraction, DatasetProxy, RecommendationResult
│   └── tests/                  # Unit and integration tests
│
├── postman/                    # Postman collections and environments
├── .env                        # Environment variables (not committed)
├── pyproject.toml              # Project metadata and dependencies
└── manage.py                   # Django management entry point
```

---

## Configuration Reference

### Django REST Framework

```python
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",   # → 401 on missing auth
        "rest_framework.authentication.SessionAuthentication", # → browser sessions
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}
```

### Celery

```python
CELERY_BROKER_URL        = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND    = "redis://localhost:6379/0"
CELERY_TASK_ALWAYS_EAGER = True   # local dev only — False in production
```

### LLM

```python
LLM_BACKEND     = "gemini"      # or "openai" | "anthropic"
LLM_MODEL       = "gemini-1.5-pro"
LLM_BATCH_SIZE  = 10            # columns per request
LLM_MAX_TOKENS  = 1024
LLM_TEMPERATURE = 0.2
```

### Recommendations

```python
RECOMMENDATIONS_ALPHA = 0.5      # default hybrid weight
CF_MODEL_TYPE         = "als"    # or "svd"
MODEL_STORE_BACKEND   = "local"  # or "s3"
```

---

## Key Design Decisions

### `metadata`

- **Pipeline is async-first** — `pipeline.run()` is never called synchronously inside a view; always dispatched via Celery. The view returns 202 immediately.
- **Adapters are pure ingestion** — no business logic; each returns a raw `pd.DataFrame`.
- **LLM calls are batched** — multiple columns are combined in a single prompt to minimise API usage and cost.
- **Credentials are never stored** — `source_config` holds env var names only, never raw secrets.
- **Missing classifier model** — if `metadata/ml_models/semantic_classifier.pkl` is absent, the semantic classifier falls back to regex-based heuristics and logs a warning.

### `recommendations`

- **`persistence.py` is the ORM boundary** — domain engines never import from `models.py` directly; all DB access flows through `infrastructure/persistence.py`.
- **Ranking is pure** — `rank()` and `mmr_rerank()` have no DB calls, no side effects, and no cache access.
- **Signals are non-blocking** — all heavy work (cache invalidation, re-scoring) is dispatched to Celery tasks, never run synchronously in a signal receiver. If the broker is unavailable, the error is logged and the save completes normally.
- **Cold-start is handled gracefully** — new users with no interaction history receive content-based popularity scores instead of empty results.
- **Cache degrades gracefully** — if Redis is unavailable, the cache layer logs a warning and the recommendation view falls back to a live engine call.
- **Scores of `0.0` are expected before training** — run `train_collaborative` and `train_content_based` to produce non-trivial scores.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `django >= 6.0` | Web framework |
| `djangorestframework >= 3.16` | REST API layer |
| `celery >= 5.6` | Async task queue |
| `pandas >= 3.0` | Data ingestion and profiling |
| `scikit-learn >= 1.8` | Semantic classifier (ML) |
| `scipy >= 1.17` | Sparse TF-IDF matrix storage |
| `numpy >= 2.4` | Numerical scoring |
| `joblib >= 1.5` | Model serialisation |
| `openpyxl / xlrd` | Excel ingestion |
| `sqlalchemy >= 2.0` | SQL database ingestion |
| `google-genai` | Gemini LLM backend |
| `openai` | OpenAI LLM backend |
| `anthropic` | Anthropic LLM backend |
| `boto3` | AWS S3 model/vector storage |
| `python-dotenv` | `.env` loading |

---

## License

This project is private. All rights reserved.
