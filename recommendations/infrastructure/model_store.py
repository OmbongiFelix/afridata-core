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

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any

import joblib
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured  # top-level: always available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelNotFoundError(FileNotFoundError):
    """
    Raised when a model weights file cannot be located at the given path.

    Callers (collaborative.py) should catch this and surface a clear error
    rather than silently returning zero scores.
    """


# ---------------------------------------------------------------------------
# Safety guard — prevent accidental storage of raw interaction data
# ---------------------------------------------------------------------------

# Attribute names that suggest the object carries raw user/interaction data
# rather than fitted model weights.  Checked in save_model() as a lightweight
# safeguard; not an exhaustive validation.
_RAW_DATA_ATTRS = frozenset(
    {
        "user_ids",
        "item_ids",
        "interaction_matrix",
        "ratings_matrix",
        "raw_interactions",
    }
)


def _assert_no_raw_data(model: Any) -> None:
    """
    Raise ``ValueError`` if *model* looks like it contains raw interaction data
    or user identifiers rather than fitted model weights.

    This is a best-effort heuristic, not a deep inspection. Its purpose is to
    catch obvious mistakes (e.g. accidentally passing a DataFrame of user events
    instead of a fitted Pipeline).

    Raises
    ------
    ValueError
        If any of the sentinel attribute names are found on *model*.
    """
    found = _RAW_DATA_ATTRS.intersection(dir(model))
    if found:
        raise ValueError(
            f"save_model() received an object that appears to contain raw "
            f"interaction data or user IDs (suspicious attributes: {sorted(found)}). "
            "Only pass fitted model weight objects — never raw interaction data."
        )


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _backend() -> str:
    """Return the configured storage backend slug, defaulting to 'local'."""
    return getattr(settings, "MODEL_STORE_BACKEND", "local").lower()


# --- Local filesystem -------------------------------------------------------


def _save_local(model: Any, path: str) -> None:
    """Serialise *model* to *path* on the local filesystem using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    logger.debug("model_store: saved to local path '%s'", path)


def _load_local(path: str) -> Any:
    """Deserialise and return the model at *path* from the local filesystem."""
    if not os.path.exists(path):
        raise ModelNotFoundError(
            f"Model weights not found at local path '{path}'. "
            "Run train_collaborative before starting inference."
        )
    model = joblib.load(path)
    logger.debug("model_store: loaded from local path '%s'", path)
    return model


# --- S3 / object store ------------------------------------------------------


def _s3_client():
    """
    Return a boto3 S3 client configured from Django settings.

    Expected settings (all optional — boto3 falls back to env vars /
    instance profile if omitted):

      AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY
      AWS_S3_ENDPOINT_URL   — set for non-AWS S3-compatible stores (e.g. MinIO)
      AWS_S3_REGION_NAME
    """
    import boto3  # local import — boto3 is an optional production dependency

    kwargs: dict[str, str] = {}
    for attr, kwarg in [
        ("AWS_ACCESS_KEY_ID",     "aws_access_key_id"),
        ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
        ("AWS_S3_ENDPOINT_URL",   "endpoint_url"),
        ("AWS_S3_REGION_NAME",    "region_name"),
    ]:
        value = getattr(settings, attr, None)
        if value:
            kwargs[kwarg] = value

    return boto3.client("s3", **kwargs)


def _s3_bucket() -> str:
    """Return the S3 bucket name from settings, raising clearly if absent."""
    bucket = getattr(settings, "MODEL_STORE_S3_BUCKET", None)
    if not bucket:
        raise ImproperlyConfigured(
            "settings.MODEL_STORE_S3_BUCKET must be set when "
            "MODEL_STORE_BACKEND = 's3'."
        )
    return bucket


def _save_s3(model: Any, path: str) -> None:
    """
    Serialise *model* into an in-memory buffer and upload to S3.

    *path* is used as the S3 object key.
    """
    buf = BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)

    client = _s3_client()
    bucket = _s3_bucket()
    client.upload_fileobj(buf, bucket, path)
    logger.debug("model_store: saved to s3://%s/%s", bucket, path)


def _load_s3(path: str) -> Any:
    """
    Download the model object at *path* from S3 and deserialise it.

    Raises
    ------
    ModelNotFoundError
        If the object does not exist in the bucket.
    """
    import botocore.exceptions  # local import alongside boto3

    client = _s3_client()
    bucket = _s3_bucket()
    buf = BytesIO()

    try:
        client.download_fileobj(bucket, path, buf)
    except botocore.exceptions.ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey"):
            raise ModelNotFoundError(
                f"Model weights not found at s3://{bucket}/{path}. "
                "Run train_collaborative before starting inference."
            ) from exc
        raise  # re-raise unexpected S3 errors

    buf.seek(0)
    model = joblib.load(buf)
    logger.debug("model_store: loaded from s3://%s/%s", bucket, path)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str) -> None:
    """
    Serialise a fitted model object and write it to the configured backend.

    Parameters
    ----------
    model:
        Any joblib-serialisable object — typically an sklearn Pipeline,
        TruncatedSVD, or implicit ALS model.  Raw interaction data or objects
        carrying user IDs must never be passed here.
    path:
        Destination path (local filesystem path or S3 object key,
        depending on ``settings.MODEL_STORE_BACKEND``).

    Raises
    ------
    ValueError
        If *model* appears to contain raw interaction data or user identifiers.
    ImproperlyConfigured
        If ``MODEL_STORE_BACKEND = 's3'`` but ``MODEL_STORE_S3_BUCKET``
        is not set.
    """
    _assert_no_raw_data(model)

    backend = _backend()
    logger.info("model_store.save_model: backend=%s path=%s", backend, path)

    if backend == "s3":
        _save_s3(model, path)
    else:
        if backend != "local":
            logger.warning(
                "model_store: unknown backend '%s', falling back to 'local'",
                backend,
            )
        _save_local(model, path)


def load_model(path: str) -> Any:
    """
    Load and return a previously saved model from the configured backend.

    Parameters
    ----------
    path:
        Source path (local filesystem path or S3 object key).

    Returns
    -------
    object
        The deserialised model, ready for inference.

    Raises
    ------
    ModelNotFoundError
        If no weights file exists at *path*.
    ImproperlyConfigured
        If ``MODEL_STORE_BACKEND = 's3'`` but ``MODEL_STORE_S3_BUCKET``
        is not set.
    """
    backend = _backend()
    logger.info("model_store.load_model: backend=%s path=%s", backend, path)

    if backend == "s3":
        return _load_s3(path)

    if backend != "local":
        logger.warning(
            "model_store: unknown backend '%s', falling back to 'local'",
            backend,
        )
    return _load_local(path)