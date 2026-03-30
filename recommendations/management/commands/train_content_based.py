"""
Management command: python manage.py train_content_based

Fetches all Dataset records from persistence.py, concatenates title,
description, and tags into a text corpus, fits a TF-IDF vectoriser,
and saves the resulting item matrix via infrastructure/vector_store.py.

Options:
  --max-features  TF-IDF vocabulary size (default: 10000)
  --ngram-range   N-gram range, e.g. '1,2' for unigrams+bigrams (default: 1,1)
  --min-df        Minimum document frequency for vocabulary inclusion (default: 1)
  --sublinear-tf  Apply sublinear TF scaling — 1 + log(tf) (default: off)
  --output        Override default matrix save path

Incremental updates are not supported — always a full rebuild.
Re-run whenever Dataset metadata is bulk-updated.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from sklearn.feature_extraction.text import TfidfVectorizer

from recommendations.infrastructure.persistence import get_all_datasets
from recommendations.infrastructure.vector_store import (
    VectorStoreError,
    save_tfidf_matrix,
)

logger = logging.getLogger(__name__)

# Placeholder injected for datasets whose title, description, and tags are all
# blank.  A non-empty string is required so TF-IDF produces a real (albeit
# low-signal) vector rather than a silent zero-vector that would permanently
# exclude the item from content-based recommendations.
_EMPTY_DOC_PLACEHOLDER = "__no_content__"

# Default save path — can be overridden via --output or settings.
_DEFAULT_MATRIX_PATH_SETTING = "TFIDF_MATRIX_PATH"
_DEFAULT_MATRIX_PATH_FALLBACK = os.path.join("ml_models", "tfidf", "content_matrix")


def _default_output_path() -> str:
    return getattr(
        settings,
        _DEFAULT_MATRIX_PATH_SETTING,
        _DEFAULT_MATRIX_PATH_FALLBACK,
    )


def _build_corpus(datasets) -> tuple[list[str], list[int]]:
    """
    Iterate over a DatasetProxy QuerySet and build parallel lists of
    document strings and dataset IDs.

    Each document concatenates (in order):
      1. title           — highest signal, repeated to up-weight it
      2. description     — free-text body
      3. tags            — space-joined tag strings for keyword matching

    Fields are gracefully skipped when blank or ``None``.  If all three
    fields are empty the document is replaced with ``_EMPTY_DOC_PLACEHOLDER``
    so that TF-IDF still produces a real (low-signal) vector for that item
    rather than a silent zero-vector.

    Parameters
    ----------
    datasets:
        QuerySet[DatasetProxy] returned by ``get_all_datasets()``.

    Returns
    -------
    corpus : list[str]
        One document string per dataset row.
    item_ids : list[int]
        Parallel list of ``DatasetProxy.dataset_id`` values; row *i* of
        the fitted TF-IDF matrix corresponds to ``item_ids[i]``.
    """
    corpus: list[str] = []
    item_ids: list[int] = []

    for dataset in datasets:
        parts: list[str] = []

        title = getattr(dataset, "title", None) or ""
        if title:
            # Repeat title to give it extra weight in the TF-IDF space.
            parts.append(f"{title} {title}")

        description = getattr(dataset, "description", None) or ""
        if description:
            parts.append(description)

        # Tags may be stored as a JSON list, a comma-separated string, or a
        # related manager.  Handle all common representations defensively.
        tags = getattr(dataset, "tags", None)
        if tags is not None:
            if hasattr(tags, "all"):
                # ManyToMany / related manager
                tag_str = " ".join(
                    str(getattr(t, "name", t)) for t in tags.all()
                )
            elif isinstance(tags, (list, tuple)):
                tag_str = " ".join(str(t) for t in tags)
            else:
                tag_str = str(tags)

            if tag_str.strip():
                parts.append(tag_str)

        document = " ".join(parts).strip()

        # Substitute placeholder so the item is never silently excluded from
        # recommendations due to a zero-vector.
        if not document:
            document = _EMPTY_DOC_PLACEHOLDER

        corpus.append(document)
        item_ids.append(int(dataset.dataset_id))

    return corpus, item_ids


class Command(BaseCommand):
    help = (
        "Train a TF-IDF content-based model from DatasetProxy metadata "
        "and persist the resulting item matrix."
    )

    # ------------------------------------------------------------------
    # Argument definitions
    # ------------------------------------------------------------------

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--max-features",
            type=int,
            default=10_000,
            metavar="N",
            help=(
                "Maximum size of the TF-IDF vocabulary.  "
                "Larger values capture more rare terms at the cost of memory.  "
                "Default: 10000."
            ),
        )
        parser.add_argument(
            "--ngram-range",
            type=str,
            default="1,1",
            metavar="MIN,MAX",
            help=(
                "Comma-separated n-gram range passed to TfidfVectorizer, "
                "e.g. '1,2' for unigrams and bigrams.  Default: '1,1'."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Override the default matrix save path.  "
                f"Defaults to settings.{_DEFAULT_MATRIX_PATH_SETTING} "
                f"or '{_DEFAULT_MATRIX_PATH_FALLBACK}'."
            ),
        )
        parser.add_argument(
            "--min-df",
            type=int,
            default=1,
            metavar="N",
            help=(
                "Minimum document frequency for a term to be included in the "
                "vocabulary.  Increase to prune very rare terms.  Default: 1."
            ),
        )
        parser.add_argument(
            "--sublinear-tf",
            action="store_true",
            default=False,
            help=(
                "Apply sublinear TF scaling (1 + log(tf)) to dampen the "
                "influence of very high-frequency terms.  Default: off."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_ngram_range(self, raw: str) -> tuple[int, int]:
        """Parse a 'MIN,MAX' string into a (min, max) int tuple."""
        try:
            parts = raw.split(",")
            if len(parts) != 2:
                raise ValueError("Expected exactly two comma-separated integers.")
            lo, hi = int(parts[0].strip()), int(parts[1].strip())
            if lo < 1 or hi < lo:
                raise ValueError("Require 1 ≤ min ≤ max.")
            return lo, hi
        except ValueError as exc:
            raise CommandError(
                f"Invalid --ngram-range '{raw}': {exc}  "
                "Example valid value: '1,2'."
            ) from exc

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(self, *args, **options) -> None:
        max_features: int = options["max_features"]
        ngram_range: tuple[int, int] = self._parse_ngram_range(options["ngram_range"])
        output_path: str = options["output"] or _default_output_path()
        min_df: int = options["min_df"]
        sublinear_tf: bool = options["sublinear_tf"]

        self.stdout.write(
            self.style.MIGRATE_HEADING("=== train_content_based ===")
        )
        self.stdout.write(f"  max_features : {max_features}")
        self.stdout.write(f"  ngram_range  : {ngram_range}")
        self.stdout.write(f"  min_df       : {min_df}")
        self.stdout.write(f"  sublinear_tf : {sublinear_tf}")
        self.stdout.write(f"  output       : {output_path}")
        self.stdout.write("")

        # ----------------------------------------------------------------
        # 1. Load datasets
        # ----------------------------------------------------------------
        self.stdout.write("Step 1/4  Fetching dataset metadata …")
        t0 = time.perf_counter()

        datasets = get_all_datasets()  # lazy QuerySet
        corpus, item_ids = _build_corpus(datasets)

        n_items = len(corpus)
        elapsed = time.perf_counter() - t0
        self.stdout.write(
            self.style.SUCCESS(
                f"         Loaded {n_items} active datasets in {elapsed:.2f}s."
            )
        )

        if n_items == 0:
            raise CommandError(
                "No active DatasetProxy records found.  "
                "Sync datasets before running this command."
            )

        # Warn about datasets that had no text content and received a
        # placeholder.  They will still be vectorised but will produce
        # low-signal vectors and are unlikely to surface in recommendations.
        placeholder_count = sum(
            1 for doc in corpus if doc == _EMPTY_DOC_PLACEHOLDER
        )
        if placeholder_count:
            self.stderr.write(
                self.style.WARNING(
                    f"WARNING: {placeholder_count} dataset(s) have no text "
                    "content (empty title, description, and tags).  "
                    f"A '{_EMPTY_DOC_PLACEHOLDER}' placeholder has been "
                    "substituted so they remain in the index, but they are "
                    "unlikely to be recommended via content-based similarity."
                )
            )

        # ----------------------------------------------------------------
        # 2. Fit TF-IDF
        # ----------------------------------------------------------------
        self.stdout.write("Step 2/4  Fitting TF-IDF vectoriser …")
        t1 = time.perf_counter()

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",  # tokens of ≥ 2 chars
            norm="l2",
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except Exception as exc:
            raise CommandError(
                f"TF-IDF vectorisation failed: {exc}"
            ) from exc

        elapsed = time.perf_counter() - t1
        vocab_size = len(vectorizer.vocabulary_)
        self.stdout.write(
            self.style.SUCCESS(
                f"         Fitted in {elapsed:.2f}s.  "
                f"Matrix shape: {tfidf_matrix.shape}  |  "
                f"Vocabulary size: {vocab_size:,}"
            )
        )

        # ----------------------------------------------------------------
        # 3. Build item_ids array
        # ----------------------------------------------------------------
        self.stdout.write("Step 3/4  Building item_id index array …")
        item_ids_array = np.array(item_ids, dtype=np.int64)
        self.stdout.write(
            self.style.SUCCESS(
                f"         item_ids array: shape={item_ids_array.shape}, "
                f"dtype={item_ids_array.dtype}"
            )
        )

        # ----------------------------------------------------------------
        # 4. Persist matrix + index
        # ----------------------------------------------------------------
        self.stdout.write(f"Step 4/4  Saving matrix to '{output_path}' …")
        t3 = time.perf_counter()

        try:
            save_tfidf_matrix(tfidf_matrix, item_ids_array, output_path)
        except VectorStoreError as exc:
            raise CommandError(
                f"Failed to save TF-IDF matrix: {exc}"
            ) from exc

        elapsed = time.perf_counter() - t3
        self.stdout.write(
            self.style.SUCCESS(
                f"         Saved in {elapsed:.2f}s."
            )
        )

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"✓ Content-based model trained successfully.  "
                f"{n_items} items × {vocab_size:,} features."
            )
        )
        logger.info(
            "train_content_based: complete — items=%d features=%d path=%s",
            n_items,
            vocab_size,
            output_path,
        )