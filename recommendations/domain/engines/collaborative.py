"""
Collaborative Filtering engine using Matrix Factorisation.

Loads pre-trained model weights from infrastructure/model_store.py
and scores a list of candidate items for a given user.

Algorithm: Alternating Least Squares (ALS) or truncated SVD.
           Configured via settings.CF_MODEL_TYPE.

Cold-start handling:
  Users with no interaction history receive uniform zero S_CF scores.
  The hybrid engine then falls back entirely to S_CBF for cold users.

Does not train. Training is handled by:
  management/commands/train_collaborative.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from django.conf import settings

from recommendations.domain.schemas import CandidateSet
from recommendations.infrastructure.model_store import ModelNotFoundError, load_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Supported model type slugs (must match what train_collaborative.py writes).
CF_MODEL_TYPE_ALS = "als"
CF_MODEL_TYPE_SVD = "svd"

# Default path used during both training and inference.
DEFAULT_MODEL_PATH: str = "models/collaborative/cf_model.joblib"

# Setting name used to select algorithm at runtime.
_CF_MODEL_TYPE_SETTING = "CF_MODEL_TYPE"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CollaborativeEngineError(RuntimeError):
    """Base class for unrecoverable errors in the collaborative filtering engine."""


class ModelNotLoadedError(CollaborativeEngineError):
    """
    Raised when model weights cannot be located or the engine is used before
    ``load()`` has been called successfully.

    Wraps ``ModelNotFoundError`` from ``model_store`` as ``__cause__`` so
    callers can inspect the underlying I/O error if needed.
    """


# ---------------------------------------------------------------------------
# Model-type helpers
# ---------------------------------------------------------------------------


def _model_type() -> str:
    """Return the configured CF model type slug, defaulting to 'als'."""
    return getattr(settings, _CF_MODEL_TYPE_SETTING, CF_MODEL_TYPE_ALS).lower()


# ---------------------------------------------------------------------------
# Scoring backends
# ---------------------------------------------------------------------------


def _score_als(
    model: Any,
    user_id: int,
    candidate_item_ids: list[int],
    item_id_to_index: dict[int, int],
) -> dict[int, float]:
    """
    Score candidate items for *user_id* using an implicit ALS model.

    The implicit library's ``AlternatingLeastSquares`` exposes
    ``model.user_factors`` and ``model.item_factors`` as numpy arrays.

    Scores are the raw dot product of the user latent vector and each
    candidate item latent vector.  They are min-max normalised to [0, 1]
    before returning so they are comparable with S_CBF scores.

    Parameters
    ----------
    model:
        A fitted ``implicit.als.AlternatingLeastSquares`` instance.
    user_id:
        Internal user index (must match the index used during training).
    candidate_item_ids:
        Dataset IDs to score.
    item_id_to_index:
        Mapping of dataset_id → row index in ``model.item_factors``.

    Returns
    -------
    dict[int, float]
        Mapping of dataset_id → normalised score in [0.0, 1.0].
        Items not found in ``item_id_to_index`` receive 0.0.
    """
    try:
        user_vector: np.ndarray = model.user_factors[user_id]  # (n_factors,)
    except IndexError:
        logger.warning(
            "collaborative._score_als: user_id=%d not in model (n_users=%d); "
            "returning zero scores",
            user_id,
            len(model.user_factors),
        )
        return {item_id: 0.0 for item_id in candidate_item_ids}

    raw: dict[int, float] = {}
    for item_id in candidate_item_ids:
        idx = item_id_to_index.get(item_id)
        if idx is None:
            raw[item_id] = 0.0
        else:
            item_vector: np.ndarray = model.item_factors[idx]  # (n_factors,)
            raw[item_id] = float(np.dot(user_vector, item_vector))

    return _minmax_normalise(raw)


def _score_svd(
    model: Any,
    user_id: int,
    candidate_item_ids: list[int],
    item_id_to_index: dict[int, int],
) -> dict[int, float]:
    """
    Score candidate items for *user_id* using a truncated SVD model.

    Expects the serialised object to be a dict with keys:
      ``user_factors``  — np.ndarray of shape (n_users, n_components)
      ``item_factors``  — np.ndarray of shape (n_items, n_components)

    This matches the convention used by
    ``management/commands/train_collaborative.py`` for SVD.

    Parameters
    ----------
    model:
        Dict with ``user_factors`` and ``item_factors`` arrays.
    user_id:
        Internal user index.
    candidate_item_ids:
        Dataset IDs to score.
    item_id_to_index:
        Mapping of dataset_id → row index in ``item_factors``.

    Returns
    -------
    dict[int, float]
        Mapping of dataset_id → normalised score in [0.0, 1.0].
    """
    user_factors: np.ndarray = model["user_factors"]
    item_factors: np.ndarray = model["item_factors"]

    if user_id >= len(user_factors):
        logger.warning(
            "collaborative._score_svd: user_id=%d not in model (n_users=%d); "
            "returning zero scores",
            user_id,
            len(user_factors),
        )
        return {item_id: 0.0 for item_id in candidate_item_ids}

    user_vector: np.ndarray = user_factors[user_id]  # (n_components,)

    raw: dict[int, float] = {}
    for item_id in candidate_item_ids:
        idx = item_id_to_index.get(item_id)
        if idx is None:
            raw[item_id] = 0.0
        else:
            raw[item_id] = float(np.dot(user_vector, item_factors[idx]))

    return _minmax_normalise(raw)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _minmax_normalise(scores: dict[int, float]) -> dict[int, float]:
    """
    Min-max normalise a ``{item_id: score}`` dict to [0.0, 1.0].

    Items that are already 0.0 (not found in the model) are excluded from
    the min/max calculation so they remain at 0.0 after normalisation — a
    score of 0.0 retains its meaning as "no data".

    If all non-zero scores are identical, they are mapped to 1.0.
    """
    nonzero = {k: v for k, v in scores.items() if v != 0.0}
    if not nonzero:
        return dict(scores)

    min_val = min(nonzero.values())
    max_val = max(nonzero.values())

    if max_val == min_val:
        return {k: (1.0 if k in nonzero else 0.0) for k in scores}

    span = max_val - min_val
    return {
        k: (0.0 if k not in nonzero else (v - min_val) / span)
        for k, v in scores.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CollaborativeEngine:
    """
    Stateful collaborative filtering recommendation engine.

    Load once per process (e.g. in AppConfig.ready) and reuse across
    requests; the model weights are held in memory.

    The active algorithm is selected by ``settings.CF_MODEL_TYPE``:
      ``"als"`` — implicit ALS (default)
      ``"svd"`` — truncated SVD

    Parameters
    ----------
    model_path:
        Path / S3 key for the serialised model weights.
        Defaults to ``DEFAULT_MODEL_PATH``.

    Examples
    --------
    >>> engine = CollaborativeEngine()
    >>> engine.load()
    >>> # Primary interface — pass a CandidateSet from the retrieval stage:
    >>> scores = engine.score(user_id=42, candidates=candidate_set)
    >>> # Lower-level interface — pass raw lists directly:
    >>> scores = engine.score_for_user(
    ...     user_id=42,
    ...     candidate_item_ids=[101, 202, 303],
    ...     item_id_to_index={101: 0, 202: 1, 303: 2},
    ... )
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: Any = None
        self._model_type: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load model weights from the configured backend into memory.

        Safe to call multiple times; subsequent calls reload from storage
        (useful for hot-reloading after retraining without restarting the
        process).

        Raises
        ------
        ModelNotLoadedError
            Wraps ``ModelNotFoundError`` if the weights file cannot be found.
        """
        resolved_type = _model_type()
        logger.info(
            "collaborative.CollaborativeEngine.load: "
            "model_type=%s path=%s",
            resolved_type,
            self._model_path,
        )
        try:
            self._model = load_model(self._model_path)
        except ModelNotFoundError as exc:
            raise ModelNotLoadedError(
                f"Cannot load collaborative model from '{self._model_path}'. "
                "Run train_collaborative to build the model first."
            ) from exc

        self._model_type = resolved_type
        logger.info(
            "collaborative: model loaded — type=%s", self._model_type
        )

    @property
    def is_loaded(self) -> bool:
        """True if the model has been successfully loaded."""
        return self._model is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "CollaborativeEngine has not been loaded. Call engine.load() first."
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, user_id: int, candidates: CandidateSet) -> dict[int, float]:
        """
        Score a ``CandidateSet`` for a single user.  Primary public interface.

        Delegates to ``score_for_user`` after unpacking the ``CandidateSet``
        so callers in the orchestrating layer never deal with raw index dicts.

        Parameters
        ----------
        user_id:
            Internal integer user index (as used during training).
        candidates:
            ``CandidateSet`` produced by the retrieval stage.  Must expose
            ``item_ids`` (list[int]) and ``item_id_to_index`` (dict[int, int]).

        Returns
        -------
        dict[int, float]
            Mapping of ``dataset_id → S_CF score`` in [0.0, 1.0].

        Raises
        ------
        ModelNotLoadedError
            If the engine has not been loaded or the weights file is missing.
        CollaborativeEngineError
            If the configured model type is unrecognised.
        """
        return self.score_for_user(
            user_id=user_id,
            candidate_item_ids=candidates.item_ids,
            item_id_to_index=candidates.item_id_to_index,
        )

    def score_for_user(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        item_id_to_index: dict[int, int],
    ) -> dict[int, float]:
        """
        Score a set of candidate items for a single user.

        Cold-start users (user_id not present in the model) receive
        uniform 0.0 scores for all candidates.  The hybrid engine detects
        this and falls back entirely to S_CBF scores.

        Parameters
        ----------
        user_id:
            Internal integer user index (as used during training).
        candidate_item_ids:
            Dataset IDs to score.
        item_id_to_index:
            Mapping of dataset_id → row index in the item factor matrix.
            Items absent from this dict receive a score of 0.0.

        Returns
        -------
        dict[int, float]
            Mapping of ``dataset_id → score`` for every item in
            ``candidate_item_ids``.  Scores are in [0.0, 1.0].

        Raises
        ------
        CollaborativeEngineError
            If the engine has not been loaded or the model type is
            unrecognised.
        """
        self._require_loaded()

        model_type = self._model_type or _model_type()

        if model_type == CF_MODEL_TYPE_ALS:
            return _score_als(
                model=self._model,
                user_id=user_id,
                candidate_item_ids=candidate_item_ids,
                item_id_to_index=item_id_to_index,
            )
        elif model_type == CF_MODEL_TYPE_SVD:
            return _score_svd(
                model=self._model,
                user_id=user_id,
                candidate_item_ids=candidate_item_ids,
                item_id_to_index=item_id_to_index,
            )
        else:
            raise CollaborativeEngineError(
                f"Unrecognised CF_MODEL_TYPE '{model_type}'. "
                f"Supported values: '{CF_MODEL_TYPE_ALS}', '{CF_MODEL_TYPE_SVD}'."
            )

    def score_batch(
        self,
        user_ids: list[int],
        candidate_item_ids: list[int],
        item_id_to_index: dict[int, int],
    ) -> dict[int, dict[int, float]]:
        """
        Score candidates for multiple users in a single pass.

        Avoids re-checking model state on every call; the model is loaded
        once and reused across all users.  Suitable for offline batch
        inference jobs.

        Parameters
        ----------
        user_ids:
            Internal user indices to score.
        candidate_item_ids:
            Dataset IDs to score for every user.
        item_id_to_index:
            Mapping of dataset_id → item factor row index.

        Returns
        -------
        dict[int, dict[int, float]]
            Mapping of ``user_id → {dataset_id: score}``.

        Raises
        ------
        CollaborativeEngineError
            If the engine has not been loaded.
        """
        self._require_loaded()

        results: dict[int, dict[int, float]] = {}
        for user_id in user_ids:
            results[user_id] = self.score_for_user(
                user_id=user_id,
                candidate_item_ids=candidate_item_ids,
                item_id_to_index=item_id_to_index,
            )
        return results

    def is_cold_start(self, scores: dict[int, float]) -> bool:
        """
        Return True if *scores* indicates a cold-start user.

        A cold-start user has all-zero scores because their user_id was
        not found in the model's factor matrices.  The hybrid engine uses
        this to decide whether to suppress S_CF entirely.

        Parameters
        ----------
        scores:
            Output of ``score_for_user`` for a single user.

        Returns
        -------
        bool
        """
        return all(v == 0.0 for v in scores.values())