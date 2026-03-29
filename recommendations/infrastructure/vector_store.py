"""
TF-IDF matrix and item vector storage for the content-based engine.

Handles persistence of scipy sparse matrices produced by the TF-IDF
vectoriser, together with their associated item_id index arrays.

Separate from model_store.py because:
  - TF-IDF matrices are scipy.sparse (not joblib-optimal)
  - They require an item_id index array stored alongside the matrix
  - They can be 100MB+ and need streaming load for memory efficiency

Used by:
  management/commands/train_content_based.py  → save_tfidf_matrix()
  domain/engines/content_based.py             → load_tfidf_matrix()
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any

import numpy as np
import scipy.sparse
from django.conf import settings

logger = logging.getLogger(__name__)

__all__ = [
    "VectorStoreError",
    "save_tfidf_matrix",
    "load_tfidf_matrix",
    "save_item_vectors",
    "load_item_vectors",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VectorStoreError(IOError):
    """
    Raised when a required matrix or index file cannot be located or loaded.

    Wraps the original exception as ``__cause__`` so callers can inspect
    the underlying I/O error if needed.
    """


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
# Each logical store consists of two sibling files that share a base path:
#   <base>.npz   — the scipy sparse matrix (save_npz / load_npz)
#   <base>.npy   — the item_id index array  (np.save / np.load)
# For dense item vectors a single <base>.npy file is used.


def _matrix_path(base: str) -> str:
    return base if base.endswith(".npz") else f"{base}.npz"


def _index_path(base: str) -> str:
    # Strip .npz extension if present so both files share the same stem.
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}_item_ids.npy"


def _vectors_path(base: str) -> str:
    return base if base.endswith(".npy") else f"{base}.npy"


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _backend() -> str:
    """Return the configured storage backend slug, defaulting to 'local'."""
    return getattr(settings, "MODEL_STORE_BACKEND", "local").lower()


# --- Local filesystem -------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _save_sparse_local(matrix: scipy.sparse.csr_matrix, path: str) -> None:
    _ensure_dir(path)
    scipy.sparse.save_npz(path, matrix)
    logger.debug("vector_store: sparse matrix saved to '%s'", path)


def _load_sparse_local(path: str) -> scipy.sparse.csr_matrix:
    if not os.path.exists(path):
        raise VectorStoreError(
            f"TF-IDF matrix file not found at '{path}'. "
            "Run train_content_based to rebuild the matrix."
        )
    matrix = scipy.sparse.load_npz(path)
    logger.debug("vector_store: sparse matrix loaded from '%s'", path)
    return matrix


def _save_array_local(array: np.ndarray, path: str) -> None:
    _ensure_dir(path)
    np.save(path, array, allow_pickle=False)
    logger.debug("vector_store: array saved to '%s'", path)


def _load_array_local(path: str) -> np.ndarray:
    # np.save appends .npy automatically; resolve the real path before
    # opening so the error message always names the file that is missing.
    if os.path.exists(path):
        resolved = path
    elif path.endswith(".npy") is False and os.path.exists(f"{path}.npy"):
        resolved = f"{path}.npy"
    else:
        raise VectorStoreError(
            f"Item ID index file not found at '{path}'. "
            "Run train_content_based to rebuild the matrix."
        )
    array = np.load(resolved, allow_pickle=False)
    logger.debug("vector_store: array loaded from '%s'", resolved)
    return array


# --- S3 / object store ------------------------------------------------------


def _s3_client() -> Any:
    """Return a boto3 S3 client, mirroring the pattern from model_store.py."""
    import boto3  # optional production dependency

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
    bucket = getattr(settings, "MODEL_STORE_S3_BUCKET", None)
    if not bucket:
        from django.core.exceptions import ImproperlyConfigured
        raise ImproperlyConfigured(
            "settings.MODEL_STORE_S3_BUCKET must be set when "
            "MODEL_STORE_BACKEND = 's3'."
        )
    return bucket


def _upload_bytes(buf: BytesIO, key: str) -> None:
    buf.seek(0)
    client = _s3_client()
    bucket = _s3_bucket()
    client.upload_fileobj(buf, bucket, key)
    logger.debug("vector_store: uploaded s3://%s/%s", bucket, key)


def _download_bytes(key: str) -> BytesIO:
    import botocore.exceptions

    client = _s3_client()
    bucket = _s3_bucket()
    buf = BytesIO()
    try:
        client.download_fileobj(bucket, key, buf)
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            raise VectorStoreError(
                f"Object not found at s3://{bucket}/{key}. "
                "Run train_content_based to rebuild the matrix."
            ) from exc
        raise VectorStoreError(
            f"S3 client error fetching s3://{bucket}/{key}: {exc}"
        ) from exc
    except botocore.exceptions.BotoCoreError as exc:
        # Covers connection errors, timeouts, and other low-level failures
        # that are not ClientError subclasses.
        raise VectorStoreError(
            f"S3 transport error fetching s3://{bucket}/{key}: {exc}"
        ) from exc
    buf.seek(0)
    logger.debug("vector_store: downloaded s3://%s/%s", bucket, key)
    return buf


def _save_sparse_s3(matrix: scipy.sparse.csr_matrix, key: str) -> None:
    buf = BytesIO()
    scipy.sparse.save_npz(buf, matrix)
    _upload_bytes(buf, key)


def _load_sparse_s3(key: str) -> scipy.sparse.csr_matrix:
    buf = _download_bytes(key)
    return scipy.sparse.load_npz(buf)


def _save_array_s3(array: np.ndarray, key: str) -> None:
    buf = BytesIO()
    np.save(buf, array, allow_pickle=False)
    _upload_bytes(buf, key)


def _load_array_s3(key: str) -> np.ndarray:
    buf = _download_bytes(key)
    return np.load(buf, allow_pickle=False)


# ---------------------------------------------------------------------------
# Public API — TF-IDF sparse matrix
# ---------------------------------------------------------------------------


def save_tfidf_matrix(
    matrix: scipy.sparse.csr_matrix,
    item_ids: np.ndarray,
    path: str,
) -> None:
    """
    Persist a TF-IDF item matrix and its accompanying item_id index.

    Two sibling files are always written together:
      ``<path>.npz``          — the sparse matrix (scipy ``save_npz``)
      ``<path>_item_ids.npy`` — the item_id index array (``np.save``)

    Parameters
    ----------
    matrix:
        Sparse TF-IDF matrix of shape ``(n_items, n_features)`` produced
        by ``sklearn.feature_extraction.text.TfidfVectorizer``.
    item_ids:
        1-D integer array of length ``n_items`` mapping row indices back
        to ``DatasetProxy.dataset_id`` values.
    path:
        Base path (local) or S3 object key prefix.  The ``.npz`` /
        ``_item_ids.npy`` suffixes are appended automatically.

    Raises
    ------
    VectorStoreError
        If writing fails for any reason.
    """
    if item_ids.ndim != 1:
        raise VectorStoreError(
            f"item_ids must be a 1-D array; got shape {item_ids.shape}."
        )
    if matrix.shape[0] != len(item_ids):
        raise VectorStoreError(
            f"matrix row count ({matrix.shape[0]}) does not match "
            f"item_ids length ({len(item_ids)})."
        )

    backend = _backend()
    matrix_key = _matrix_path(path)
    index_key = _index_path(path)

    logger.info(
        "vector_store.save_tfidf_matrix: backend=%s shape=%s path=%s",
        backend, matrix.shape, path,
    )

    try:
        if backend == "s3":
            _save_sparse_s3(matrix, matrix_key)
            _save_array_s3(item_ids, index_key)
        else:
            if backend != "local":
                logger.warning(
                    "vector_store: unknown backend '%s', falling back to 'local'",
                    backend,
                )
            _save_sparse_local(matrix, matrix_key)
            _save_array_local(item_ids, index_key)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to save TF-IDF matrix to '{path}': {exc}"
        ) from exc


def load_tfidf_matrix(
    path: str,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Load a previously saved TF-IDF matrix and its item_id index.

    Both sibling files must be present.  Loading only one is an error
    because a matrix without its item_id mapping is useless for scoring.

    Parameters
    ----------
    path:
        Base path (local) or S3 object key prefix used when saving.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, np.ndarray]
        ``(matrix, item_ids)`` — the sparse TF-IDF matrix and the
        parallel integer array of ``DatasetProxy.dataset_id`` values.

    Raises
    ------
    VectorStoreError
        If either file is missing, cannot be deserialised, or the loaded
        matrix and index array have incompatible lengths.
    """
    backend = _backend()
    matrix_key = _matrix_path(path)
    index_key = _index_path(path)

    logger.info(
        "vector_store.load_tfidf_matrix: backend=%s path=%s", backend, path
    )

    try:
        if backend == "s3":
            matrix = _load_sparse_s3(matrix_key)
            item_ids = _load_array_s3(index_key)
        else:
            if backend != "local":
                logger.warning(
                    "vector_store: unknown backend '%s', falling back to 'local'",
                    backend,
                )
            matrix = _load_sparse_local(matrix_key)
            item_ids = _load_array_local(index_key)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to load TF-IDF matrix from '{path}': {exc}"
        ) from exc

    # Guard against a stale index file that no longer matches the matrix,
    # e.g. if the matrix was retrained but the index write was interrupted.
    if matrix.shape[0] != len(item_ids):
        raise VectorStoreError(
            f"Loaded matrix row count ({matrix.shape[0]}) does not match "
            f"item_ids length ({len(item_ids)}) for path '{path}'. "
            "The store may be corrupt — re-run train_content_based."
        )

    logger.info(
        "vector_store: loaded matrix shape=%s item_ids=%d",
        matrix.shape, len(item_ids),
    )
    return matrix, item_ids


# ---------------------------------------------------------------------------
# Public API — dense item vectors (embedding alternative)
# ---------------------------------------------------------------------------


def save_item_vectors(vectors: np.ndarray, path: str) -> None:
    """
    Persist a dense item embedding matrix to the configured backend.

    Stored as a single ``.npy`` file.  Use this when the content engine
    is switched from TF-IDF cosine similarity to a dense embedding model.

    Parameters
    ----------
    vectors:
        2-D float array of shape ``(n_items, embedding_dim)``.
    path:
        Destination path / S3 key.  A ``.npy`` suffix is appended if absent.

    Raises
    ------
    VectorStoreError
        If writing fails.
    """
    if vectors.ndim != 2:
        raise VectorStoreError(
            f"vectors must be a 2-D array; got shape {vectors.shape}."
        )

    backend = _backend()
    key = _vectors_path(path)

    logger.info(
        "vector_store.save_item_vectors: backend=%s shape=%s path=%s",
        backend, vectors.shape, key,
    )

    try:
        if backend == "s3":
            _save_array_s3(vectors, key)
        else:
            if backend != "local":
                logger.warning(
                    "vector_store: unknown backend '%s', falling back to 'local'",
                    backend,
                )
            _save_array_local(vectors, key)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to save item vectors to '{path}': {exc}"
        ) from exc


def load_item_vectors(path: str) -> np.ndarray:
    """
    Load a dense item embedding matrix from the configured backend.

    Parameters
    ----------
    path:
        Source path / S3 key used when saving.

    Returns
    -------
    np.ndarray
        2-D float array of shape ``(n_items, embedding_dim)``.

    Raises
    ------
    VectorStoreError
        If the file is missing, cannot be deserialised, or the loaded
        array is not 2-D.
    """
    backend = _backend()
    key = _vectors_path(path)

    logger.info(
        "vector_store.load_item_vectors: backend=%s path=%s", backend, key
    )

    try:
        if backend == "s3":
            vectors = _load_array_s3(key)
        else:
            if backend != "local":
                logger.warning(
                    "vector_store: unknown backend '%s', falling back to 'local'",
                    backend,
                )
            vectors = _load_array_local(key)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to load item vectors from '{path}': {exc}"
        ) from exc

    if vectors.ndim != 2:
        raise VectorStoreError(
            f"Expected a 2-D array from '{path}'; got shape {vectors.shape}. "
            "The file may be corrupt or was saved by a different version."
        )

    logger.info(
        "vector_store: loaded item vectors shape=%s", vectors.shape
    )
    return vectors