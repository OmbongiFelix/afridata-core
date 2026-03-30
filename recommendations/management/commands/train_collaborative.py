"""
Management command: python manage.py train_collaborative

Fetches the full UserInteraction history from persistence.py,
builds a user-item interaction matrix, fits a Matrix Factorisation
model (ALS or SVD), and saves the trained weights via
infrastructure/model_store.py.

Options:
  --factors   Number of latent factors (default: 50)
  --epochs    Training iterations (default: 20)
  --algo      Algorithm to use: 'svd' or 'als' (default: 'svd')
  --output    Override default model save path
  --evaluate  Run evaluation metrics against a held-out test split
  --test-size Fraction of interactions to hold out for evaluation (default: 0.2)

Run nightly via Celery beat or after any bulk interaction import.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from recommendations.domain.evaluation import evaluate_engine
from recommendations.infrastructure.model_store import save_model
from recommendations.infrastructure.persistence import get_user_interactions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FACTORS = 50
DEFAULT_EPOCHS = 20
DEFAULT_TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InteractionRecord:
    """Flat representation of a single user-item interaction."""

    user_id: int
    item_id: int
    weight: float  # implicit feedback signal (e.g. 1.0 per interaction)


@dataclass
class TrainedModel:
    """
    Container for all artefacts produced by the training run.

    Saved as a single joblib blob so that collaborative.py can load
    everything it needs with one ``load_model`` call.
    """

    # Core matrix factorisation components
    user_factors: np.ndarray          # shape (n_users, n_factors)
    item_factors: np.ndarray          # shape (n_items, n_factors)

    # Label encoders — map raw DB ids ↔ matrix row/col indices
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder

    # Metadata
    algorithm: str
    n_factors: int
    n_epochs: int
    trained_at: float = field(default_factory=time.time)

    # Optional: the full interaction matrix (kept for potential online updates)
    interaction_matrix: csr_matrix | None = None


# ---------------------------------------------------------------------------
# Interaction fetching
# ---------------------------------------------------------------------------


def _fetch_all_interactions() -> list[InteractionRecord]:
    """
    Pull every UserInteraction from the DB via persistence helpers and
    return a flat list of InteractionRecord objects.

    Each DB row becomes one record with weight=1.0 (binary implicit
    feedback).  Callers can aggregate by (user, item) to get
    interaction counts if desired.
    """
    from recommendations.models import UserInteraction  # local import avoids circular refs

    # We need all interactions across all users — iterate over all user ids
    # that appear in the table.  Using values_list is cheaper than loading
    # full ORM objects.
    raw_qs = (
        UserInteraction.objects
        .values_list("user_id", "dataset_id")
        .order_by("user_id")
    )

    records: list[InteractionRecord] = []
    for user_id, dataset_id in raw_qs.iterator(chunk_size=2000):
        records.append(
            InteractionRecord(user_id=user_id, item_id=dataset_id, weight=1.0)
        )

    return records


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def _build_interaction_matrix(
    records: list[InteractionRecord],
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
) -> csr_matrix:
    """
    Convert a list of InteractionRecord objects into a sparse CSR matrix.

    Duplicate (user, item) pairs are *summed* — a user who viewed the
    same dataset multiple times gets a higher implicit feedback weight.

    Parameters
    ----------
    records:
        Output of ``_fetch_all_interactions()``.
    user_encoder:
        Fitted LabelEncoder mapping user_id → row index.
    item_encoder:
        Fitted LabelEncoder mapping item_id → col index.

    Returns
    -------
    csr_matrix
        Shape (n_users, n_items), dtype float32.
    """
    user_indices = user_encoder.transform([r.user_id for r in records])
    item_indices = item_encoder.transform([r.item_id for r in records])
    weights = np.array([r.weight for r in records], dtype=np.float32)

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)

    matrix = csr_matrix(
        (weights, (user_indices, item_indices)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    # sum_duplicates merges repeated (row, col) entries
    matrix.sum_duplicates()
    return matrix


# ---------------------------------------------------------------------------
# Training algorithms
# ---------------------------------------------------------------------------


def _train_svd(
    matrix: csr_matrix,
    n_factors: int,
    n_epochs: int,
    stdout,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit TruncatedSVD (randomised SVD) on the interaction matrix.

    SVD decomposes  M ≈ U · S · Vᵀ.  We absorb the singular values
    into U to get user factors, and use Vᵀ directly as item factors.

    Returns
    -------
    user_factors : np.ndarray  shape (n_users, n_factors)
    item_factors : np.ndarray  shape (n_items, n_factors)
    loss         : float       reconstruction loss (1 - explained variance ratio sum)
    """
    stdout.write(
        f"  Fitting TruncatedSVD: n_components={n_factors}, "
        f"n_iter={n_epochs} (random SVD iterations) …"
    )
    svd = TruncatedSVD(n_components=n_factors, n_iter=n_epochs, random_state=42)
    user_factors = svd.fit_transform(matrix)           # (n_users, n_factors)
    item_factors = svd.components_.T                   # (n_items, n_factors)

    explained = svd.explained_variance_ratio_.sum()
    loss = 1.0 - explained
    stdout.write(f"  SVD explained variance: {explained:.2%}  |  Final loss (unexplained variance): {loss:.4f}")
    return user_factors, item_factors, loss


def _train_als(
    matrix: csr_matrix,
    n_factors: int,
    n_epochs: int,
    stdout,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit an implicit ALS model via the ``implicit`` library.

    Falls back gracefully if ``implicit`` is not installed, raising a
    clear ``CommandError`` so the operator knows what to install.

    Returns
    -------
    user_factors : np.ndarray  shape (n_users, n_factors)
    item_factors : np.ndarray  shape (n_items, n_factors)
    loss         : float       final training loss reported by the ALS model
    """
    try:
        import implicit  # optional production dependency
    except ImportError as exc:
        raise CommandError(
            "The 'implicit' package is required for ALS training. "
            "Install it with:  pip install implicit"
        ) from exc

    stdout.write(
        f"  Fitting ALS: factors={n_factors}, iterations={n_epochs} …"
    )
    # implicit expects items × users
    item_user_matrix = matrix.T.tocsr().astype(np.float32)

    model = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        iterations=n_epochs,
        use_gpu=False,
        random_state=42,
        calculate_training_loss=True,
    )
    model.fit(item_user_matrix)

    # implicit stores factors as (n_items, factors) and (n_users, factors)
    item_factors = model.item_factors   # np.ndarray
    user_factors = model.user_factors   # np.ndarray

    # Retrieve the last recorded training loss if available
    loss: float = float("nan")
    if hasattr(model, "loss_") and model.loss_:
        loss = float(model.loss_[-1])
    elif hasattr(model, "loss"):
        loss = float(model.loss)
    stdout.write(f"  ALS final training loss: {loss:.4f}")

    return user_factors, item_factors, loss


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------


class Command(BaseCommand):
    help = (
        "Train a Matrix Factorisation collaborative filtering model and "
        "save the weights to the configured model store."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--factors",
            type=int,
            default=DEFAULT_FACTORS,
            metavar="N",
            help=f"Number of latent factors (default: {DEFAULT_FACTORS}).",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=DEFAULT_EPOCHS,
            metavar="N",
            help=f"Number of training iterations (default: {DEFAULT_EPOCHS}).",
        )
        parser.add_argument(
            "--algo",
            choices=["svd", "als"],
            default="svd",
            help="Matrix factorisation algorithm: 'svd' (default) or 'als'.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Override the default model save path. "
                "Defaults to settings.CF_MODEL_PATH if not provided."
            ),
        )
        parser.add_argument(
            "--evaluate",
            action="store_true",
            default=False,
            help="Hold out a test split and report Precision@10 / NDCG@10 after training.",
        )
        parser.add_argument(
            "--test-size",
            type=float,
            default=DEFAULT_TEST_SIZE,
            metavar="FRAC",
            help=(
                f"Fraction of interactions to use as the test split when "
                f"--evaluate is set (default: {DEFAULT_TEST_SIZE})."
            ),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, *args, **options) -> None:  # noqa: ARG002
        n_factors: int = options["factors"]
        n_epochs: int = options["epochs"]
        algo: str = options["algo"]
        output_path: str = options["output"] or settings.CF_MODEL_PATH
        do_evaluate: bool = options["evaluate"]
        test_size: float = options["test_size"]

        self.stdout.write(self.style.MIGRATE_HEADING("=== train_collaborative ==="))
        self.stdout.write(
            f"  algorithm={algo}  factors={n_factors}  epochs={n_epochs}"
        )
        self.stdout.write(f"  output path: {output_path}")

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Fetch interactions
        # ------------------------------------------------------------------
        self.stdout.write("\n[1/4] Fetching interactions from database …")
        t_fetch = time.perf_counter()

        all_records = _fetch_all_interactions()

        if not all_records:
            raise CommandError(
                "No UserInteraction records found in the database. "
                "Import some user data before running this command."
            )

        self.stdout.write(
            f"  Loaded {len(all_records):,} interaction records in "
            f"{time.perf_counter() - t_fetch:.1f}s."
        )

        # ------------------------------------------------------------------
        # 2. Encode users & items; optionally split train / test
        # ------------------------------------------------------------------
        self.stdout.write("\n[2/4] Encoding users and items …")

        all_user_ids = sorted({r.user_id for r in all_records})
        all_item_ids = sorted({r.item_id for r in all_records})

        user_encoder = LabelEncoder().fit(all_user_ids)
        item_encoder = LabelEncoder().fit(all_item_ids)

        self.stdout.write(
            f"  {len(all_user_ids):,} unique users  ×  "
            f"{len(all_item_ids):,} unique items."
        )

        train_records = all_records
        test_records: list[InteractionRecord] = []

        if do_evaluate:
            if not (0.0 < test_size < 1.0):
                raise CommandError(
                    f"--test-size must be between 0 and 1, got {test_size}."
                )
            train_records, test_records = train_test_split(
                all_records, test_size=test_size, random_state=42
            )
            self.stdout.write(
                f"  Train: {len(train_records):,}  Test: {len(test_records):,} "
                f"({test_size:.0%} held out)."
            )

        # ------------------------------------------------------------------
        # 3. Build sparse matrix and fit model
        # ------------------------------------------------------------------
        self.stdout.write("\n[3/4] Building interaction matrix and training …")
        t_train = time.perf_counter()

        interaction_matrix = _build_interaction_matrix(
            train_records, user_encoder, item_encoder
        )
        self.stdout.write(
            f"  Matrix shape: {interaction_matrix.shape}  "
            f"density: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}"
        )

        if algo == "als":
            user_factors, item_factors, final_loss = _train_als(
                interaction_matrix, n_factors, n_epochs, self.stdout
            )
        else:
            user_factors, item_factors, final_loss = _train_svd(
                interaction_matrix, n_factors, n_epochs, self.stdout
            )

        elapsed_train = time.perf_counter() - t_train
        self.stdout.write(
            f"  Training complete in {elapsed_train:.1f}s.  Final loss: {final_loss:.4f}"
        )

        # ------------------------------------------------------------------
        # 4. (Optional) Evaluate via domain evaluate_engine
        # ------------------------------------------------------------------
        if do_evaluate and test_records:
            self.stdout.write("\n[4a/4] Evaluating on held-out test set …")

            # Build a temporary model snapshot to pass into evaluate_engine
            eval_model = TrainedModel(
                user_factors=user_factors.astype(np.float32),
                item_factors=item_factors.astype(np.float32),
                user_encoder=user_encoder,
                item_encoder=item_encoder,
                algorithm=algo,
                n_factors=n_factors,
                n_epochs=n_epochs,
            )

            metrics: dict[str, Any] = evaluate_engine(
                model=eval_model,
                test_interactions=test_records,
                top_k=10,
            )

            precision = metrics.get("precision_at_10", metrics.get("precision_at_k", float("nan")))
            ndcg = metrics.get("ndcg_at_10", metrics.get("ndcg_at_k", float("nan")))

            self.stdout.write(
                self.style.SUCCESS(
                    f"  Precision@10: {precision:.4f}  |  NDCG@10: {ndcg:.4f}  "
                    f"(n_test_users={metrics.get('n_test_users', '?')})"
                )
            )
            logger.info(
                "Evaluation complete — Precision@10=%.4f  NDCG@10=%.4f",
                precision,
                ndcg,
            )

        # ------------------------------------------------------------------
        # 5. Persist model
        # ------------------------------------------------------------------
        self.stdout.write("\n[4/4] Saving model weights …")

        trained_model = TrainedModel(
            user_factors=user_factors.astype(np.float32),
            item_factors=item_factors.astype(np.float32),
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            algorithm=algo,
            n_factors=n_factors,
            n_epochs=n_epochs,
            interaction_matrix=interaction_matrix,
        )

        save_model(trained_model, output_path)

        total_elapsed = time.perf_counter() - t0
        self.stdout.write(
            self.style.SUCCESS(
                f"\n✓ Model saved to '{output_path}'.  "
                f"Total elapsed: {total_elapsed:.1f}s."
            )
        )
        logger.info(
            "train_collaborative complete — algo=%s factors=%d epochs=%d "
            "loss=%.4f elapsed=%.1fs path=%s",
            algo,
            n_factors,
            n_epochs,
            final_loss,
            total_elapsed,
            output_path,
        )