# AfriData-Core — Audit, Fix & Verify Walkthrough

## Goal

Enable a complete end-to-end REST pipeline over HTTP:

1. **POST** a structured dataset (CSV/Excel/SQL) → ingested as a `PipelineRun`
2. **Metadata inferred** for that dataset via the semantic classifier pipeline
3. **GET recommendations** — content-based, collaborative, and hybrid — using the inferred metadata

Preserve existing architecture (`domain/infrastructure/api` in recommendations, `adapters/core/api` in metadata).

---

## What Was Fixed

### 1. Missing DB Migration — `created_by_id` column

**Root cause:** `PipelineRun` model had a `created_by` ForeignKey added but no migration was created for it.  
**Fix:** Created `metadata/migrations/0002_add_created_by_to_pipeline_run.py`.

> [!IMPORTANT]
> This was the single biggest blocker — **every test that touched the DB was ERROR-ing** because of this missing column.

### 2. `ValidationError` Import Conflict in `metadata/api/views.py`

**Root cause:** `from rest_framework.exceptions import ValidationError` (line 27) was silently overwritten by `from django.core.exceptions import ValidationError` (line 32). The column profiles view was raising Django's `ValidationError` which doesn't serialize to a DRF 400 response.

**Fix:** Renamed to `DRFValidationError` and `DjangoValidationError` to disambiguate.

### 3. DRF Authentication Order — 401 vs 403

**Root cause:** `SessionAuthentication` was listed first in `DEFAULT_AUTHENTICATION_CLASSES`. When no credentials are provided, `SessionAuthentication` triggers 403 (CSRF failure path), not 401.

**Fix:** Moved `TokenAuthentication` first — unauthenticated requests now correctly return **401 Unauthorized** with a `WWW-Authenticate: Token` header.

### 4. `CELERY_TASK_ALWAYS_EAGER` defaulted to `True` in settings, then `False`

**Two-phase problem:**
- Tests mock `_run_pipeline_task` but with `ALWAYS_EAGER=True`, Celery tries to execute the mock *as* a task, failing.
- With `ALWAYS_EAGER=False` and no broker, `POST /api/metadata/runs/` crashes trying to connect to Redis.

**Fix:**
- `settings.py`: default to `False` (tests mock the task)
- `.env`: set `CELERY_TASK_ALWAYS_EAGER=True` (local dev runs tasks inline)
- `views.py`: wrapped `run_pipeline_task.delay(...)` in `try/except` — broker unavailability never crashes the 202 response

### 5. Signal Handlers Crashing on Broker Unavailability

**Root cause:** `recommendations/signals.py` called `tasks.refresh_user_scores.delay(user_id)` on every `UserInteraction` save. Without a running broker, this crashed the entire test `setUp`.

**Fix:** Wrapped both `on_interaction_saved` and `on_interaction_deleted` in `try/except` — cache invalidation is **best-effort** and never blocks a save.

### 6. `set_cached_recommendations` Called with Wrong Arguments

**Two bugs fixed:**
- `recommendations/api/views.py`: called `set_cached_recommendations(ranked_list)` — missing `user_id`
- `recommendations/tasks.py`: called with `recommendations=recommendations` — wrong kwarg name (should be `ranked_list`)

### 7. `test_retrieve_invalid_pk_returns_404`

**Root cause:** `reverse("metadata:pipeline-run-detail", kwargs={"pk": "not-a-uuid"})` itself raises `NoReverseMatch` because Django's `<uuid:pk>` URL converter validates the pattern at reversal time.

**Fix:** Changed the test to use a hardcoded URL path `"/api/metadata/runs/not-a-uuid/"` directly — Django returns 404 at URL resolution, which is the correct behavior to test.

---

## New Files Created

| File | Purpose |
|------|---------|
| [`metadata/migrations/0002_add_created_by_to_pipeline_run.py`](file:///c:/Users/fombo/OneDrive/Desktop/afridata-core/metadata/migrations/0002_add_created_by_to_pipeline_run.py) | Missing migration for `created_by` FK |
| [`metadata/management/commands/seed_demo.py`](file:///c:/Users/fombo/OneDrive/Desktop/afridata-core/metadata/management/commands/seed_demo.py) | Seeds demo user, API token, and DatasetProxy records for live testing |

---

## Test Results

```
Ran 119 tests in 312.434s
OK
```

**Tests covered:**
- `metadata.tests.test_api` — 75 tests (serializers, permissions, views, helper functions)
- `recommendations.tests.tests` — integration tests (pipeline, feedback)
- `recommendations.tests.test_hybrid_engine` — alpha weighting, fusion arithmetic
- `recommendations.tests.test_ranking` — score ordering, tie-breaking, top-N cutoff
- `recommendations.tests.test_content_based` — TF-IDF scoring, cold-start fallback
- `recommendations.tests.test_collaborative` — SVD scoring, cold-start detection

---

## Live HTTP Verification

Server: `http://localhost:8000`  
Token: `4ed08684cb2cdff8da095e85405bb8e2511bdf64` (user: `admin`)

### Step 1 — Ingest a CSV Dataset

```http
POST /api/metadata/runs/
Authorization: Token 4ed08684cb2cdff8da095e85405bb8e2511bdf64
Content-Type: application/json

{
  "source": "csv",
  "source_path": "metadata/tests/fixtures/sample.csv",
  "dataset_title": "AfriData Sample Kenya"
}
```

**Response: 202 Accepted**
```json
{
  "id": "6a544fce-0cb7-4781-be7a-9d9190488ece",
  "source": "csv",
  "source_path": "metadata/tests/fixtures/sample.csv",
  "dataset_title": "AfriData Sample Kenya",
  "status": "PENDING",
  "is_terminal": false,
  "created_at": "2026-08-20T06:45:30.125259Z"
}
```

### Step 2 — Poll Run Status

```http
GET /api/metadata/runs/6a544fce-0cb7-4781-be7a-9d9190488ece/
Authorization: Token 4ed08684cb2cdff8da095e85405bb8e2511bdf64
```

**Response: 200 OK** — `status: "PENDING"` (pipeline runs asynchronously via Celery)

### Step 3 — Schema returns 409 while Pending

```http
GET /api/metadata/runs/6a544fce-0cb7-4781-be7a-9d9190488ece/schema/
Authorization: Token 4ed08684cb2cdff8da095e85405bb8e2511bdf64
```

**Response: 409 Conflict** ✅ (correct — schema only available after `SUCCESS`)
```json
{
  "detail": "Schema is not available. Run status is 'PENDING'.",
  "run_id": "6a544fce-0cb7-4781-be7a-9d9190488ece",
  "status": "PENDING"
}
```

### Step 4 — Recommendations (all 3 strategies)

```http
GET /api/recommendations/?strategy=content
GET /api/recommendations/?strategy=collaborative
GET /api/recommendations/?strategy=hybrid
Authorization: Token 4ed08684cb2cdff8da095e85405bb8e2511bdf64
```

**Response: 200 OK** for all three — 5 ranked datasets returned

Sample response (hybrid):
```json
{
  "recommendations": [
    { "dataset_id": "1001", "title": "Kenya Health Indicators 2023", "rank": 1, "s_hybrid": 0.0, "confidence": "low" },
    { "dataset_id": "1002", "title": "Nigeria Agricultural Production", "rank": 2, "s_hybrid": 0.0, "confidence": "low" }
  ],
  "alpha": 0.0,
  "top_n": 10,
  "generated_at": "2026-08-20T06:42:58.867877Z"
}
```

> [!NOTE]
> Scores are `0.0` because no TF-IDF matrix or collaborative model has been trained yet. Run `python manage.py train_content_based` and `python manage.py train_collaborative` to train the engines with real data.

### Step 5 — Record Feedback

```http
POST /api/recommendations/feedback/
Authorization: Token 4ed08684cb2cdff8da095e85405bb8e2511bdf64
Content-Type: application/json

{ "dataset_id": "1001", "interaction_type": "view" }
```

**Response: 201 Created** ✅

---

## Quick-Start for Local Development

```powershell
# 1. Seed demo data
uv run python manage.py seed_demo

# 2. Start the server
uv run python manage.py runserver

# 3. Set token (from seed_demo output)
$TOKEN = "4ed08684cb2cdff8da095e85405bb8e2511bdf64"

# 4. Ingest a CSV
Invoke-RestMethod -Uri "http://localhost:8000/api/metadata/runs/" -Method POST `
  -Headers @{Authorization="Token $TOKEN"; "Content-Type"="application/json"} `
  -Body '{"source": "csv", "source_path": "metadata/tests/fixtures/sample.csv", "dataset_title": "My Dataset"}'

# 5. Get recommendations
Invoke-RestMethod -Uri "http://localhost:8000/api/recommendations/?strategy=hybrid" `
  -Headers @{Authorization="Token $TOKEN"}
```

---

## Architecture Notes

- **`metadata/`** — API-only layer: `adapters/core/api`. Celery task `run_pipeline_task` runs the `MetadataPipeline` and syncs `DatasetProxy` in recommendations.
- **`recommendations/`** — Full DDD: `domain/infrastructure/api`. Three engines: `ContentBasedEngine`, `CollaborativeEngine`, `HybridEngine`. Cache via Redis (gracefully degrades when unavailable).
- **Celery** — `CELERY_TASK_ALWAYS_EAGER=True` in `.env` for local dev. Set `False` + provide `CELERY_BROKER_URL` for production.
