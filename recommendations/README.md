# Recommendations Engine

A Django application (`recommendations/`) that delivers personalised dataset recommendations using a weighted hybrid of Collaborative Filtering (CF) and Content-Based Filtering (CBF), served via a REST API with Redis caching and Celery async processing.

---

## Recommendation Flow

```
User Data → Candidate Gen → Collaborative (SCF) + Content-Based (SCBF) → Hybrid Fusion → Ranking → Top-N → API Response
Stage 1      Stage 2         Stage 3a                Stage 3b             Stage 4         Stage 5   Stage 6   Stage 7
```

| Stage | Module | Description |
|-------|--------|-------------|
| 1 – User Data | `models.py` | `UserInteraction` records provide the training signal |
| 2 – Candidate Gen | `engines/candidate_gen.py` | Build the eligible item pool, excluding seen items |
| 3a – Collaborative | `engines/collaborative.py` | Score via Matrix Factorisation (ALS/SVD) → **S_CF** |
| 3b – Content-Based | `engines/content_based.py` | Score via TF-IDF cosine similarity → **S_CBF** |
| 4 – Hybrid Fusion | `engines/hybrid.py` | `S_hybrid = α · S_CF + (1 − α) · S_CBF` |
| 5 – Ranking | `domain/ranking.py` | Sort by S_hybrid, apply Top-N, optional MMR diversity |
| 6 – Top-N | `infrastructure/cache.py` | Write ranked list to Redis (`rec:user:{user_id}`) |
| 7 – API Response | `api/views.py` | Serve from cache or trigger live engine call |

---

## Project Structure

```
recommendations/
├── models.py                          # ORM models: UserInteraction, Dataset, RecommendationResult
├── signals.py                         # post_save/post_delete → cache invalidation via Celery
├── tasks.py                           # Celery tasks: refresh_user_scores, train_*
├── apps.py                            # AppConfig — connects signals in ready()
├── admin.py                           # Django admin for monitoring runs
├── domain/
│   ├── schemas.py                     # Shared dataclasses: CandidateSet, ScoredCandidate, RankedList, EngineConfig
│   ├── evaluation.py                  # Offline metrics: Precision@K, Recall@K, NDCG@K
│   ├── ranking.py                     # Post-fusion sort, Top-N cutoff, MMR diversity re-ranking
│   └── engines/
│       ├── candidate_gen.py           # Stage 2: filter seen items, return CandidateSet
│       ├── collaborative.py           # Stage 3a: load model weights, return S_CF scores
│       ├── content_based.py           # Stage 3b: TF-IDF cosine similarity, return S_CBF scores
│       └── hybrid.py                  # Stage 4: fuse scores, delegate to ranking.py
├── infrastructure/
│   ├── persistence.py                 # All ORM query helpers — only file that imports models.py
│   ├── model_store.py                 # Save/load collaborative filter weights (joblib / S3)
│   ├── vector_store.py                # Save/load TF-IDF sparse matrices (scipy / S3)
│   └── cache.py                       # Redis helpers: get, set, invalidate per-user cache
├── api/
│   ├── views.py                       # DRF views: GET recommendations, POST feedback
│   ├── serializers.py                 # DRF serializers for all API resources
│   └── urls.py                        # URL routing — include in project urls.py
├── management/commands/
│   ├── train_collaborative.py         # Fit Matrix Factorisation model, save weights
│   ├── train_content_based.py         # Build TF-IDF matrix from Dataset metadata
│   └── rebuild_index.py               # Invalidate all caches, recompute Top-N for all users
└── tests/
    ├── tests.py                       # Integration tests: full pipeline + API endpoints
    ├── test_candidate_generation.py   # Unit tests: seen-item filtering, cold-start, pool cap
    ├── test_ranking.py                # Unit tests: ordering, Top-N, tie-breaking, MMR
    └── test_hybrid_engine.py          # Unit tests: fusion formula, alpha edge cases
```

---

## Quick Start

### 1. Register the app

```python
# settings.py
INSTALLED_APPS = [
    ...
    "recommendations.apps.RecommendationsConfig",
]
```

### 2. Include API routes

```python
# urls.py
path("api/", include("recommendations.api.urls")),
```

### 3. Configure the hybrid weight and model paths

```python
# settings.py
RECOMMENDATIONS_ALPHA = 0.5       # 0.0 = CBF only, 1.0 = CF only
CF_MODEL_PATH = "models/cf.pkl"
CF_MODEL_TYPE = "als"             # or "svd"
MODEL_STORE_BACKEND = "local"     # or "s3"
```

### 4. Train the models

```bash
# Fit collaborative filter from interaction history
python manage.py train_collaborative --factors 50 --epochs 20 --evaluate

# Build TF-IDF matrix from Dataset metadata
python manage.py train_content_based --max-features 10000

# Propagate new models to all user caches
python manage.py rebuild_index
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/recommendations/` | Return Top-N recommended datasets for the authenticated user |
| `POST` | `/api/recommendations/feedback/` | Submit explicit feedback (rating, thumbs up/down) |

Both endpoints require `IsAuthenticated`. The GET endpoint reads from Redis cache first and falls back to a live `HybridEngine` call only on a cache miss.

### Example GET response

```json
{
  "alpha": 0.5,
  "top_n": 10,
  "generated_at": "2024-11-01T12:00:00Z",
  "results": [
    {
      "dataset_id": 42,
      "title": "UK Housing Prices 2023",
      "rank": 1,
      "s_hybrid": 0.91
    }
  ]
}
```

---

## Fusion Formula

```
S_hybrid = α · S_CF + (1 − α) · S_CBF
```

| α value | Behaviour |
|---------|-----------|
| `1.0` | Collaborative filtering only |
| `0.0` | Content-based filtering only |
| `0.5` | Equal weight (default) |

Scores are min-max normalised to `[0, 1]` before ranking.

---

## Cold-Start Handling

| Scenario | Collaborative | Content-Based |
|----------|--------------|---------------|
| New user, no interactions | Returns `S_CF = 0.0` for all items | Falls back to global popularity scores |
| All items already seen | `CandidateGenerator` returns empty `CandidateSet` — no error raised |

---

## Infrastructure

### Redis Cache

Cache key format: `rec:user:{user_id}` — default TTL 1 hour.

```python
# Programmatic invalidation
from recommendations.infrastructure.cache import invalidate_user_cache
invalidate_user_cache(user_id=123)
```

Cache is automatically invalidated when a `UserInteraction` is saved or deleted (via Django signals → Celery task).

### Model Storage Backends

| Backend | Config | Use case |
|---------|--------|----------|
| `local` | `MODEL_STORE_BACKEND = "local"` | Development |
| `s3` | `MODEL_STORE_BACKEND = "s3"` | Production |

The CF weights file (`.pkl`) and TF-IDF matrix files (`.npz` + `.npy`) are stored separately — `model_store.py` handles the former, `vector_store.py` the latter.

---

## Celery Tasks

| Task | Trigger | Description |
|------|---------|-------------|
| `refresh_user_scores(user_id)` | Signal on `UserInteraction` save/delete | Recomputes and caches Top-N for one user |
| `train_collaborative_task()` | Celery beat (nightly) or management command | Full model refit from interaction history |
| `train_content_based_task()` | After bulk Dataset metadata update | Rebuilds TF-IDF matrix |

All tasks use `autoretry_for=(Exception,)` with `max_retries=3` and exponential backoff.

---

## Management Commands

```bash
# Fit collaborative filter model
python manage.py train_collaborative [--factors 50] [--epochs 20] [--evaluate] [--output path]

# Rebuild TF-IDF content matrix
python manage.py train_content_based [--max-features 10000] [--ngram-range 1,2] [--output path]

# Invalidate and recompute all user caches
python manage.py rebuild_index [--users 1,2,3] [--alpha 0.5] [--dry-run]
```

**Typical sequence after a model update:**

```
train_collaborative → train_content_based → rebuild_index
```

---

## Diversity Re-ranking (MMR)

When `EngineConfig.diversity_weight > 0`, the ranking module applies a Maximal Marginal Relevance (MMR) variant that penalises consecutive items from the same dataset category, improving result variety without sacrificing too much relevance.

---

## Running Tests

```bash
python manage.py test recommendations
```

| Test File | Scope | External deps |
|-----------|-------|---------------|
| `tests/tests.py` | Integration: full pipeline + API | Redis mocked, Celery mocked |
| `test_candidate_generation.py` | Unit: filtering, cold-start, pool cap | All DB calls mocked |
| `test_ranking.py` | Unit: ordering, Top-N, MMR | None — pure Python |
| `test_hybrid_engine.py` | Unit: fusion formula, alpha edge cases | Both engines mocked |

Target: **80%+ coverage** on `domain/` and `infrastructure/`. Never assert on exact score values — assert on structure, ordering, and field presence.

---

## Key Design Decisions

- **`persistence.py` is the ORM boundary** — domain engines never import from `models.py` directly; all DB access flows through `infrastructure/persistence.py`.
- **`hybrid.py` owns orchestration, not logic** — keep it under 120 lines; scoring belongs in the individual engines, ordering belongs in `ranking.py`.
- **All ranking functions are pure** — `rank()` and `mmr_rerank()` have no DB calls, no side effects, and no cache access.
- **Signals never block** — all heavy work (recompute, cache write) is dispatched to Celery tasks, never run synchronously in a signal receiver.
- **Views stay thin** — if a view needs more than a cache lookup and a serialiser call, the logic belongs in the domain layer or a task.
- **`schemas.py` is the domain contract** — all inter-module types are defined here; import from here to avoid circular dependencies.
- **Training is offline only** — `collaborative.py` and `content_based.py` load and score; `fit()` logic lives exclusively in the management commands.