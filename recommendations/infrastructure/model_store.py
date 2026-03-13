"""
Model weight storage for the collaborative filtering engine.

Handles persistence of fitted sklearn / implicit ALS model objects.
Uses joblib for serialisation — safe for numpy arrays embedded in
sklearn Pipeline objects.

Storage backends (configured via settings.MODEL_STORE_BACKEND):
  local   — reads/writes to the local filesystem (default, development)
  s3      — reads/writes to an S3-compatible object store (production)

Used by:
  management/commands/train_collaborative.py  → save_model()
  domain/engines/collaborative.py             → load_model()
"""
