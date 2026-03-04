# Recommendation Engine

A multi-stage recommendation system combining **Collaborative Filtering** and **Content-Based** techniques to deliver personalized item suggestions.

**Pipeline flow:** User Data → Collaborative / Content-Based Engines → Hybrid Scoring → API Response

---

## Directory Structure

| Directory | Role |
| ----------- | ------ |
| `api/` | Presentation layer — exposes REST endpoints and serializes results |
| `domain/` | Core algorithms — recommendation engines, hybrid logic, and evaluation |
| `infrastructure/` | Data & performance layer — caching, persistence, background tasks, and signals |

---

## `api/`

The presentation layer responsible for exposing recommendation results to clients.

| File | Responsibility |
| ------ | --------------- |
| `serializers.py` | Converts complex recommendation objects into JSON-serializable formats |
| `views.py` | Exposes REST endpoints for the frontend to request user-specific feeds |

---

## `domain/`

The core computational logic and algorithmic heart of the pipeline.

### `domain/engines/`

Individual recommendation modules, each implementing a distinct strategy.

| File | Responsibility |
| ------ | --------------- |
| `collaborative.py` | Implements user-item interaction models (e.g. Matrix Factorization) |
| `content_based.py` | Suggests items by comparing item metadata similarity |
| `hybrid.py` | Orchestrates the final output by weighting or switching between collaborative and content-based scores |

### Supporting Files

| File | Responsibility |
| ------ | --------------- |
| `evaluation.py` | Measures performance metrics including Precision, Recall, and Diversity |
| `schemas.py` | Defines standard data structures shared across the calculation process |

---

## `infrastructure/`

Handles all data access, caching, and asynchronous processing needs of the engine.

| File | Responsibility |
| ----------- | --------------- |
| `cache.py` | Interfaces with Redis/Memcached to store pre-computed recommendations for low-latency retrieval |
| `persistence.py` | Optimized database queries to build training sets from user logs and item metadata |
| `models.py` | Stores user preference profiles and historical recommendation logs |
| `tasks.py` | Celery tasks for heavy offline model training and batch processing |
| `signals.py` | Triggers recommendation updates in response to real-time user actions (e.g. likes, views) |