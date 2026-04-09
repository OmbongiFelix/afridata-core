"""
Content-Based Filtering engine using TF-IDF and Cosine Similarity.

Loads a precomputed TF-IDF item matrix from infrastructure/vector_store.py
and scores candidates by cosine similarity to a user profile vector.

User profile construction:
  The profile vector is the weighted average of TF-IDF vectors for all
  items the user has interacted with. Weights are determined by
  UserInteraction.weight (download > view > implicit).

Cold-start handling:
  Users with no interactions receive scores based on global item popularity
  rather than a personal profile vector.

Does not rebuild the matrix. Rebuilding is handled by:
  management/commands/train_content_based.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize

from recommendations.infrastructure.persistence import get_user_interactions
from recommendations.infrastructure.vector_store import VectorStoreError, load_tfidf_matrix
from recommendations.domain.schemas import CandidateSet

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Default base path for the TF-IDF matrix; callers can override via
# the ``matrix_path`` argument.  Expected to match the path used during
# training in management/commands/train_content_based.py.
DEFAULT_MATRIX_PATH: str = "models/tfidf/item_matrix"

# Interaction weight constants (mirroring UserInteraction.weight values).
# Higher weight → more influence on the user profile vector.
WEIGHT_DOWNLOAD: float = 3.0
WEIGHT_VIEW: float = 1.0
WEIGHT_IMPLICIT: float = 0.5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContentEngineError(RuntimeError):
    """Raised for unrecoverable errors in the content-based engine."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_profile(
    tfidf_matrix: scipy.sparse.csr_matrix,
    item_ids: np.ndarray,
    interacted_item_ids: list[int],
    interaction_weights: list[float],
) -> scipy.sparse.csr_matrix | None:
    """
    Build a weighted-average TF-IDF profile vector for a single user.

    Parameters
    ----------
    tfidf_matrix:
        Sparse TF-IDF matrix of shape ``(n_items, n_features)``.
    item_ids:
        1-D integer array mapping row indices → ``DatasetProxy.dataset_id``.
    interacted_item_ids:
        Ordered list of dataset_id values the user has interacted with.
    interaction_weights:
        Parallel list of floating-point weights for each interaction.
        Must be the same length as ``interacted_item_ids``.

    Returns
    -------
    scipy.sparse.csr_matrix or None
        Sparse row vector of shape ``(1, n_features)``, or ``None`` if
        none of the interacted IDs are present in the matrix.
    """
    if len(interacted_item_ids) != len(interaction_weights):
        raise ValueError(
            "interacted_item_ids and interaction_weights must have the same length."
        )

    # Map dataset_id → row index in tfidf_matrix.
    id_to_row: dict[int, int] = {int(iid): idx for idx, iid in enumerate(item_ids)}

    weighted_sum: scipy.sparse.csr_matrix | None = None
    total_weight: float = 0.0

    for item_id, weight in zip(interacted_item_ids, interaction_weights):
        row_idx = id_to_row.get(int(item_id))
        if row_idx is None:
            logger.debug(
                "content_based._build_user_profile: item_id=%d not found in matrix, skipping",
                item_id,
            )
            continue
        item_vec = tfidf_matrix[row_idx]  # shape (1, n_features)
        weighted_vec = item_vec * weight
        weighted_sum = weighted_vec if weighted_sum is None else weighted_sum + weighted_vec
        total_weight += weight

    if weighted_sum is None or total_weight == 0.0:
        logger.debug(
            "content_based._build_user_profile: no matching items found; returning None"
        )
        return None

    profile = weighted_sum / total_weight  # normalise by total weight
    return profile


def _cosine_scores(
    tfidf_matrix: scipy.sparse.csr_matrix,
    profile_vector: scipy.sparse.csr_matrix,
) -> np.ndarray:
    """
    Compute cosine similarity between a user profile vector and all items.

    Both inputs are L2-normalised before the dot product so the result
    is in [-1, 1] (practically [0, 1] for non-negative TF-IDF features).

    Uses sparse matrix operations throughout to avoid memory overflow on
    large matrices (10k+ items × 10k features). This is intentionally
    preferred over ``sklearn.metrics.pairwise.cosine_similarity``, which
    materialises a dense intermediate matrix.

    Parameters
    ----------
    tfidf_matrix:
        Sparse matrix of shape ``(n_items, n_features)``.
    profile_vector:
        Sparse row vector of shape ``(1, n_features)``.

    Returns
    -------
    np.ndarray
        Dense 1-D array of shape ``(n_items,)`` with cosine similarities.
    """
    # L2-normalise rows on copies (preserve originals).
    normed_matrix = normalize(tfidf_matrix, norm="l2", copy=True)
    normed_profile = normalize(profile_vector, norm="l2", copy=True)

    # Dot product: (n_items, n_features) · (n_features, 1) → (n_items, 1)
    scores = (normed_matrix @ normed_profile.T).toarray().ravel()
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ContentBasedEngine:
    """
    Stateful content-based recommendation engine.

    Load once per process (e.g. in AppConfig.ready) and reuse across
    requests; the TF-IDF matrix is held in memory.

    The primary public entry-point is ``score()``, which resolves a user's
    interaction history automatically via ``get_user_interactions``.
    For callers that already hold interaction data (e.g. batch jobs),
    ``score_for_user()`` accepts pre-fetched lists directly.

    Parameters
    ----------
    matrix_path:
        Base path / S3 key prefix for the persisted TF-IDF matrix.
        Defaults to ``DEFAULT_MATRIX_PATH``.

    Examples
    --------
    >>> engine = ContentBasedEngine()
    >>> engine.load()

    # High-level: resolve interactions automatically
    >>> scores = engine.score(user_id=42, candidates=[101, 202, 303, 404])

    # Low-level: supply interactions directly
    >>> scores = engine.score_for_user(
    ...     interacted_item_ids=[101, 202],
    ...     interaction_weights=[3.0, 1.0],
    ...     candidate_item_ids=[101, 202, 303, 404],
    ...     item_popularities={101: 50, 202: 30, 303: 20, 404: 5},
    ... )
    """

    def __init__(self, matrix_path: str = DEFAULT_MATRIX_PATH) -> None:
        self._matrix_path = matrix_path
        self._tfidf_matrix: scipy.sparse.csr_matrix | None = None
        self._item_ids: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the TF-IDF matrix and item_id index into memory.

        Safe to call multiple times; subsequent calls reload from storage
        (useful for hot-reloading after retraining).

        Raises
        ------
        ContentEngineError
            Wraps ``VectorStoreError`` if the matrix cannot be loaded.
        """
        logger.info(
            "content_based.ContentBasedEngine.load: loading matrix from '%s'",
            self._matrix_path,
        )
        try:
            self._tfidf_matrix, self._item_ids = load_tfidf_matrix(self._matrix_path)
        except VectorStoreError as exc:
            raise ContentEngineError(
                f"Cannot load TF-IDF matrix from '{self._matrix_path}'. "
                "Run train_content_based to build the matrix first."
            ) from exc

        logger.info(
            "content_based: matrix loaded — shape=%s, n_items=%d",
            self._tfidf_matrix.shape,
            len(self._item_ids),
        )

    @property
    def is_loaded(self) -> bool:
        """True if the matrix has been successfully loaded."""
        return self._tfidf_matrix is not None and self._item_ids is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ContentEngineError(
                "ContentBasedEngine has not been loaded. Call engine.load() first."
            )

    # ------------------------------------------------------------------
    # Profile construction (public)
    # ------------------------------------------------------------------

    def build_user_profile(
        self,
        interactions: list[dict],
    ) -> scipy.sparse.csr_matrix | None:
        """
        Build and return a sparse TF-IDF profile vector for a user.

        Delegates to the module-level ``_build_user_profile`` helper after
        unpacking the interaction dicts.  Exposed publicly so callers can
        inspect or cache the profile independently of scoring.

        Parameters
        ----------
        interactions:
            List of dicts, each with keys:
              - ``item_id``  (int)   — dataset ID
              - ``weight``   (float) — interaction weight
                (use ``WEIGHT_DOWNLOAD``, ``WEIGHT_VIEW``, or ``WEIGHT_IMPLICIT``)

        Returns
        -------
        scipy.sparse.csr_matrix or None
            Sparse row vector of shape ``(1, n_features)``, or ``None`` for
            a cold-start user whose items are all absent from the matrix.

        Raises
        ------
        ContentEngineError
            If the engine has not been loaded.
        """
        self._require_loaded()
        assert self._tfidf_matrix is not None
        assert self._item_ids is not None

        item_ids = [int(i["item_id"]) for i in interactions]
        weights = [float(i["weight"]) for i in interactions]

        return _build_user_profile(
            tfidf_matrix=self._tfidf_matrix,
            item_ids=self._item_ids,
            interacted_item_ids=item_ids,
            interaction_weights=weights,
        )

    # ------------------------------------------------------------------
    # Scoring — high-level (spec entry-point)
    # ------------------------------------------------------------------

    def score(
        self,
        user_id: int,
        candidates: CandidateSet,
        item_popularities: dict[int, float] | None = None,
        exclude_interacted: bool = True,
    ) -> dict[int, float]:
        """
        Score candidate items for a user, resolving their interaction history
        automatically via ``get_user_interactions``.

        This is the primary public entry-point for online inference.
        For batch jobs that already hold interaction data, prefer
        ``score_for_user()`` to avoid redundant DB round-trips.

        Parameters
        ----------
        user_id:
            The user to score for.
        candidates:
            ``CandidateSet`` — a list of dataset IDs to rank.
        item_popularities:
            Mapping of dataset_id → raw popularity count used for
            cold-start fallback.  If ``None``, cold-start users receive
            uniform 0.0 scores.
        exclude_interacted:
            If ``True`` (default), already-interacted items receive a score
            of 0.0 so they are effectively excluded from recommendations.

        Returns
        -------
        dict[int, float]
            Mapping of ``dataset_id → S_CBF score`` in ``[0.0, 1.0]``.

        Raises
        ------
        ContentEngineError
            If the engine has not been loaded.
        """
        self._require_loaded()

        raw_interactions = get_user_interactions(user_id)
        interacted_item_ids: list[int] = [i["item_id"] for i in raw_interactions]
        interaction_weights: list[float] = [i["weight"] for i in raw_interactions]

        return self.score_for_user(
            interacted_item_ids=interacted_item_ids,
            interaction_weights=interaction_weights,
            candidate_item_ids=list(candidates),
            item_popularities=item_popularities or {},
            exclude_interacted=exclude_interacted,
        )

    # ------------------------------------------------------------------
    # Scoring — low-level (pre-fetched interactions)
    # ------------------------------------------------------------------

    def score_for_user(
        self,
        interacted_item_ids: list[int],
        interaction_weights: list[float],
        candidate_item_ids: list[int],
        item_popularities: dict[int, float],
        exclude_interacted: bool = True,
    ) -> dict[int, float]:
        """
        Score a set of candidate items for a single user.

        For users with interactions the score is the cosine similarity
        between the user's weighted TF-IDF profile vector and each
        candidate item vector.

        For cold-start users (no interactions, or none found in the matrix)
        the score falls back to normalised global popularity so that the
        caller still receives a ranked list.

        Parameters
        ----------
        interacted_item_ids:
            Dataset IDs the user has previously interacted with.
        interaction_weights:
            Parallel weight for each interaction
            (use ``WEIGHT_DOWNLOAD``, ``WEIGHT_VIEW``, or ``WEIGHT_IMPLICIT``).
        candidate_item_ids:
            Dataset IDs to score.  Items not present in the TF-IDF matrix
            receive a score of 0.0.
        item_popularities:
            Mapping of dataset_id → raw popularity count (e.g. total views).
            Used only for cold-start fallback.
        exclude_interacted:
            If ``True`` (default), already-interacted items receive a score
            of 0.0 so they are effectively excluded from recommendations.

        Returns
        -------
        dict[int, float]
            Mapping of ``dataset_id → score`` for every item in
            ``candidate_item_ids``.  Scores are in ``[0.0, 1.0]``.

        Raises
        ------
        ContentEngineError
            If the engine has not been loaded.
        """
        self._require_loaded()

        assert self._tfidf_matrix is not None  # appease type checkers
        assert self._item_ids is not None

        interacted_set = set(interacted_item_ids) if exclude_interacted else set()

        # --- build user profile ------------------------------------------
        profile = _build_user_profile(
            tfidf_matrix=self._tfidf_matrix,
            item_ids=self._item_ids,
            interacted_item_ids=interacted_item_ids,
            interaction_weights=interaction_weights,
        )

        # --- cold-start fallback -----------------------------------------
        if profile is None:
            logger.info(
                "content_based: cold-start user detected — falling back to popularity scores"
            )
            return self._popularity_scores(
                candidate_item_ids=candidate_item_ids,
                item_popularities=item_popularities,
                exclude_set=interacted_set,
            )

        # --- cosine similarity scores ------------------------------------
        all_scores: np.ndarray = _cosine_scores(self._tfidf_matrix, profile)

        # Build a lookup: dataset_id → cosine score
        id_to_score: dict[int, float] = {
            int(iid): float(all_scores[idx])
            for idx, iid in enumerate(self._item_ids)
        }

        result: dict[int, float] = {}
        for item_id in candidate_item_ids:
            if item_id in interacted_set:
                result[item_id] = 0.0
            else:
                result[item_id] = id_to_score.get(item_id, 0.0)

        logger.debug(
            "content_based.score_for_user: scored %d candidates",
            len(result),
        )
        return result

    def score_batch(
        self,
        user_interactions: list[dict],
        candidate_item_ids: list[int],
        item_popularities: dict[int, float],
        exclude_interacted: bool = True,
    ) -> dict[int, dict[int, float]]:
        """
        Score candidates for multiple users in a single pass.

        Avoids reloading the matrix between users.  Useful for offline
        batch inference jobs where interaction data is already available
        (e.g. from a pre-fetched queryset).  For online single-user scoring
        with automatic interaction resolution, use ``score()`` instead.

        Parameters
        ----------
        user_interactions:
            List of dicts, each with keys:
              - ``user_id`` (int)
              - ``item_ids`` (list[int])
              - ``weights``  (list[float])
        candidate_item_ids:
            Dataset IDs to score for every user.
        item_popularities:
            Mapping of dataset_id → popularity count.
        exclude_interacted:
            If ``True``, zero-out items the user has interacted with.

        Returns
        -------
        dict[int, dict[int, float]]
            Mapping of ``user_id → {dataset_id: score}``.
        """
        self._require_loaded()

        results: dict[int, dict[int, float]] = {}
        for user in user_interactions:
            user_id: int = user["user_id"]
            results[user_id] = self.score_for_user(
                interacted_item_ids=user.get("item_ids", []),
                interaction_weights=user.get("weights", []),
                candidate_item_ids=candidate_item_ids,
                item_popularities=item_popularities,
                exclude_interacted=exclude_interacted,
            )
        return results

    # ------------------------------------------------------------------
    # Cold-start helper
    # ------------------------------------------------------------------

    @staticmethod
    def _popularity_scores(
        candidate_item_ids: list[int],
        item_popularities: dict[int, float],
        exclude_set: set[int],
    ) -> dict[int, float]:
        """
        Return min-max normalised popularity scores for cold-start users.

        Items in ``exclude_set`` receive a score of 0.0.

        Parameters
        ----------
        candidate_item_ids:
            Items to score.
        item_popularities:
            Raw popularity counts.
        exclude_set:
            Item IDs to suppress (already-interacted items).

        Returns
        -------
        dict[int, float]
            Normalised scores in ``[0.0, 1.0]``.
        """
        raw_scores: dict[int, float] = {
            item_id: (
                0.0
                if item_id in exclude_set
                else float(item_popularities.get(item_id, 0.0))
            )
            for item_id in candidate_item_ids
        }

        max_pop = max(raw_scores.values(), default=0.0)
        if max_pop == 0.0:
            return {item_id: 0.0 for item_id in candidate_item_ids}

        return {item_id: score / max_pop for item_id, score in raw_scores.items()}