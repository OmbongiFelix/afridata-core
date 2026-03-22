"""
Database models for the recommendations app.

Three models cover the full lifecycle of a recommendation:

  UserInteraction  — one record per user action on a dataset
                     (view, download, bookmark, explicit rating).
                     Primary training signal for collaborative filtering.

  DatasetProxy     — lightweight metadata proxy kept in sync with the
                     datasets app. Provides title, description, tags,
                     and category for TF-IDF content-based scoring.

  RecommendationResult — persisted Top-N recommendation list per user.
                         Stores ranked item IDs, fused scores, the alpha
                         weight used, and generation timestamp.

Models are data containers only. No scoring or ranking logic here.
"""

from django.db import models
from django.conf import settings
from django.utils import timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class InteractionType(models.TextChoices):
    VIEW         = "view",         "View"
    DOWNLOAD     = "download",     "Download"
    BOOKMARK     = "bookmark",     "Bookmark"
    RATING       = "rating",       "Explicit Rating"
    SEARCH_CLICK = "search_click", "Search Click"


# Implicit interaction weights used during collaborative filtering training.
# Higher weight = stronger positive signal.
INTERACTION_WEIGHTS: dict[str, float] = {
    InteractionType.VIEW:         1.0,
    InteractionType.SEARCH_CLICK: 1.5,
    InteractionType.BOOKMARK:     3.0,
    InteractionType.DOWNLOAD:     5.0,
    InteractionType.RATING:       0.0,  # ratings use explicit_rating value directly
}


# ---------------------------------------------------------------------------
# UserInteraction
# ---------------------------------------------------------------------------

class UserInteraction(models.Model):
    """
    One record per user action on a dataset.

    This is the primary training signal for collaborative filtering.
    Implicit interactions (view, download, bookmark) are weighted by
    INTERACTION_WEIGHTS. Explicit ratings are stored in explicit_rating
    and used as-is when interaction_type == RATING.

    Indexes
    -------
    - (user, dataset_id)          — fast lookup of a user's full history
    - (dataset_id, interaction_type) — fast item-popularity queries
    - (created_at)                — time-windowed training slices
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendation_interactions",
        db_index=True,
    )

    # References the PK of the canonical Dataset in the datasets app.
    # No ForeignKey constraint so this app stays decoupled from datasets app.
    # on_delete behaviour is handled in signals.py (deactivate proxy on delete).
    dataset_id = models.IntegerField(
        db_index=True,
        help_text="Primary key of the dataset in the datasets app table.",
    )

    interaction_type = models.CharField(
        max_length=20,
        choices=InteractionType.choices,
        default=InteractionType.VIEW,
        db_index=True,
    )

    # Only populated when interaction_type == RATING.
    explicit_rating = models.FloatField(
        null=True,
        blank=True,
        help_text="User-supplied rating (e.g. 1–5). Null for implicit interactions.",
    )

    # Seconds the user spent on the dataset detail page.
    # Useful as an additional implicit signal (longer dwell → stronger interest).
    dwell_seconds = models.PositiveIntegerField(
        default=0,
        help_text="Seconds spent on dataset page. 0 for non-view interactions.",
    )

    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        app_label = "recommendations"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "dataset_id"]),
            models.Index(fields=["dataset_id", "interaction_type"]),
        ]
        # One record per (user, dataset, interaction_type) tuple.
        # Use update_or_create in signals/views to increment dwell_seconds instead.
        constraints = [
            models.UniqueConstraint(
                fields=["user", "dataset_id", "interaction_type"],
                name="unique_user_dataset_interaction_type",
            )
        ]

    def __str__(self) -> str:
        return (
            f"UserInteraction(user={self.user_id}, "
            f"dataset={self.dataset_id}, type={self.interaction_type})"
        )

    @property
    def implicit_weight(self) -> float:
        """Return the scalar training weight for this interaction."""
        if (
            self.interaction_type == InteractionType.RATING
            and self.explicit_rating is not None
        ):
            return float(self.explicit_rating)
        return INTERACTION_WEIGHTS.get(self.interaction_type, 1.0)


# ---------------------------------------------------------------------------
# DatasetProxy
# ---------------------------------------------------------------------------

class DatasetProxy(models.Model):
    """
    Lightweight metadata mirror of a record in the datasets app.

    Exists so the recommendations engine can run content-based scoring
    (TF-IDF, tag overlap, category match) without joining across app
    boundaries at query time. Kept in sync via a post_save signal in
    signals.py whenever the canonical Dataset record changes.

    Fields are intentionally minimal — only what the scoring engines need.
    Heavy payloads (file paths, licence text, etc.) stay in the datasets app.
    """

    # Matches the PK of the canonical Dataset in the datasets app.
    dataset_id = models.IntegerField(
        unique=True,
        db_index=True,
        help_text="PK of the source dataset record in the datasets app.",
    )

    title = models.CharField(max_length=512)

    # Short description used for TF-IDF bag-of-words scoring.
    description = models.TextField(blank=True, default="")

    # Comma-separated tag slugs, e.g. "climate,africa,csv".
    # Stored denormalised for fast in-Python set-intersection scoring.
    tags = models.TextField(
        blank=True,
        default="",
        help_text="Comma-separated tag slugs copied from the datasets app.",
    )

    category = models.CharField(
        max_length=128,
        blank=True,
        default="",
        help_text="Top-level category slug, e.g. 'health', 'finance'.",
    )

    # Organisation / owner name — used as a soft categorical feature.
    organisation = models.CharField(max_length=256, blank=True, default="")

    # Licence type as a short slug, e.g. "cc-by", "odc-pddl".
    licence = models.CharField(max_length=64, blank=True, default="")

    # File format(s) available, e.g. "csv,json".
    formats = models.CharField(max_length=128, blank=True, default="")

    # Aggregate interaction count — used as a popularity prior / fallback ranker.
    interaction_count = models.PositiveIntegerField(default=0, db_index=True)

    # Average explicit rating across all RATING interactions.
    average_rating = models.FloatField(null=True, blank=True)

    # Whether the source dataset is still publicly available.
    is_active = models.BooleanField(default=True, db_index=True)

    last_synced_at = models.DateTimeField(auto_now=True)
    created_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "recommendations"
        ordering = ["-interaction_count"]
        verbose_name = "dataset proxy"
        verbose_name_plural = "dataset proxies"

    def __str__(self) -> str:
        return f"DatasetProxy(dataset_id={self.dataset_id}, title={self.title[:60]})"

    @property
    def tag_set(self) -> set[str]:
        """Return tags as a Python set for O(1) intersection scoring."""
        return {t.strip() for t in self.tags.split(",") if t.strip()}

    @property
    def text_corpus(self) -> str:
        """
        Concatenated text field fed to the TF-IDF vectoriser.
        Title is repeated to give it higher term frequency.
        """
        return (
            f"{self.title} {self.title} "
            f"{self.description} {self.tags} {self.category}"
        )


# ---------------------------------------------------------------------------
# RecommendationResult
# ---------------------------------------------------------------------------

class RecommendationResult(models.Model):
    """
    Persisted Top-N recommendation list for a user.

    Generated by the hybrid engine (hybrid.py) and written here so the
    API can return results instantly without re-running the pipeline on
    every request. Invalidated and regenerated via signals.py whenever
    the user logs a new interaction.

    ranked_dataset_ids  — ordered list of dataset PKs, best first.
    scores              — parallel list of fused hybrid scores.
    alpha               — content-based weight used during fusion
                          (collaborative weight = 1 − alpha).
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendation_result",
        db_index=True,
    )

    # JSON array of integer dataset PKs, ordered best-first.
    # e.g. [42, 7, 101, 3, ...]
    ranked_dataset_ids = models.JSONField(
        default=list,
        help_text="Ordered list of dataset PKs. Index 0 = top recommendation.",
    )

    # Parallel array of floats corresponding to ranked_dataset_ids.
    scores = models.JSONField(
        default=list,
        help_text="Fused hybrid score for each ranked dataset ID.",
    )

    # The alpha used during hybrid fusion: final = alpha * cb_score + (1-alpha) * cf_score
    alpha = models.FloatField(
        default=0.5,
        help_text="Content-based weight in [0, 1]. 1 = pure content, 0 = pure CF.",
    )

    class EngineUsed(models.TextChoices):
        HYBRID       = "hybrid",        "Hybrid"
        CONTENT      = "content_based", "Content-Based Only"
        COLLABORATIVE= "collaborative", "Collaborative Only"
        FALLBACK     = "fallback",      "Popularity Fallback"

    engine_used = models.CharField(
        max_length=20,
        choices=EngineUsed.choices,
        default=EngineUsed.HYBRID,
    )

    # How many candidates were evaluated before ranking to Top-N.
    candidate_pool_size = models.PositiveIntegerField(default=0)

    generated_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        app_label = "recommendations"
        ordering = ["-generated_at"]

    def __str__(self) -> str:
        return (
            f"RecommendationResult(user={self.user_id}, "
            f"n={len(self.ranked_dataset_ids)}, "
            f"engine={self.engine_used}, "
            f"generated_at={self.generated_at:%Y-%m-%d %H:%M})"
        )

    @property
    def top_n(self) -> list[int]:
        """Return ranked dataset IDs as a typed list of ints."""
        return [int(pk) for pk in self.ranked_dataset_ids]

    def as_scored_pairs(self) -> list[tuple[int, float]]:
        """Return [(dataset_id, score), ...] sorted best-first."""
        return sorted(
            zip(self.top_n, self.scores),
            key=lambda x: x[1],
            reverse=True,
        )