"""
Shared data contracts for the recommendations domain layer.

All inter-module communication in domain/ uses these types.
Import from here — never import from individual engine files
to avoid circular dependencies.

Types:
    CandidateSet        — output of candidate_generation: user_id + list of item_ids
    ScoredCandidate     — one item with s_cf, s_cbf, and s_hybrid scores
    RankedList          — ordered list of ScoredCandidate up to Top-N
    EngineConfig        — runtime settings: alpha, top_n, diversity_weight, candidate_pool_size

No Django imports, no database calls. This file must be importable in
isolation for use in tests and management commands without a running
Django server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_unit_float(name: str, value: float) -> None:
    """Raise ValueError if *value* is not in the closed interval [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _validate_positive_int(name: str, value: int) -> None:
    """Raise ValueError if *value* is not a positive integer."""
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")


# ---------------------------------------------------------------------------
# CandidateSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateSet:
    """
    Output of the candidate generation stage (Stage 2).

    Carries the pool of dataset IDs that are eligible to be scored for
    a given user. Already filtered to exclude items the user has seen.

    Attributes
    ----------
    user_id:
        The user for whom candidates were generated.
    item_ids:
        Ordered list of dataset PKs eligible for scoring.
        Empty when the user has interacted with every available item.
    """

    user_id: int
    item_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError(f"user_id must be a positive integer, got {self.user_id!r}")

    @property
    def is_empty(self) -> bool:
        """True when there are no eligible candidates."""
        return len(self.item_ids) == 0

    @property
    def size(self) -> int:
        """Number of candidate items in the pool."""
        return len(self.item_ids)


# ---------------------------------------------------------------------------
# ScoredCandidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredCandidate:
    """
    A single dataset item with component and fused scores.

    Produced by the hybrid fusion step (Stage 4) after both engines
    have scored the candidate pool. Scores are normalised to [0, 1].

    Attributes
    ----------
    item_id:
        Dataset PK.
    s_cf:
        Collaborative filtering score in [0, 1].
        0.0 for cold-start users or items absent from the CF model.
    s_cbf:
        Content-based filtering score in [0, 1].
        0.0 when no user profile can be constructed (cold start).
    s_hybrid:
        Fused score: alpha * s_cf + (1 - alpha) * s_cbf, normalised.
        This is the primary sort key used by ranking.py.
    category:
        Optional category slug copied from DatasetProxy.
        Used by the MMR diversity re-ranker in ranking.py.
        Empty string when category is unavailable.
    """

    item_id: int
    s_cf: float
    s_cbf: float
    s_hybrid: float
    category: str = ""

    def __post_init__(self) -> None:
        _validate_unit_float("s_cf",     self.s_cf)
        _validate_unit_float("s_cbf",    self.s_cbf)
        _validate_unit_float("s_hybrid", self.s_hybrid)

    @property
    def is_cold_start(self) -> bool:
        """True when both component scores are zero (fully cold-start user)."""
        return self.s_cf == 0.0 and self.s_cbf == 0.0


# ---------------------------------------------------------------------------
# RankedList
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankedList:
    """
    Ordered Top-N recommendation list for a user.

    Produced by domain/ranking.py and consumed by:
    - infrastructure/cache.py  (serialisation → Redis)
    - infrastructure/persistence.py  (serialisation → RecommendationResult)
    - api/serializers.py  (shape → JSON API response)

    Attributes
    ----------
    user_id:
        The user this list was generated for.
    items:
        ScoredCandidate objects sorted by s_hybrid descending.
        Length <= EngineConfig.top_n.
    generated_at:
        UTC timestamp of when ranking.rank() produced this list.
        Defaults to the current UTC time at construction.
    engine_used:
        Which engine path produced the list. One of:
        'hybrid', 'content_based', 'collaborative', 'fallback'.
    alpha:
        The alpha value used during hybrid fusion.
        Informational — stored alongside results for auditability.
    """

    user_id: int
    items: List[ScoredCandidate] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    engine_used: str = "hybrid"
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError(f"user_id must be a positive integer, got {self.user_id!r}")
        _validate_unit_float("alpha", self.alpha)
        valid_engines = {"hybrid", "content_based", "collaborative", "fallback"}
        if self.engine_used not in valid_engines:
            raise ValueError(
                f"engine_used must be one of {valid_engines}, got {self.engine_used!r}"
            )

    @property
    def is_empty(self) -> bool:
        """True when no recommendations were produced."""
        return len(self.items) == 0

    @property
    def top_n(self) -> int:
        """Number of items in this list."""
        return len(self.items)

    @property
    def ranked_dataset_ids(self) -> List[int]:
        """Ordered list of dataset PKs, best first. Convenience for persistence layer."""
        return [item.item_id for item in self.items]

    @property
    def scores(self) -> List[float]:
        """Parallel list of s_hybrid scores. Convenience for persistence layer."""
        return [item.s_hybrid for item in self.items]

    def to_cache_dict(self) -> dict:
        """
        Serialise to a JSON-safe dict for storage in Redis via cache.py.

        Uses only built-in types (str, int, float, list, dict) so that
        json.dumps() works without a custom encoder.
        """
        return {
            "user_id":      self.user_id,
            "engine_used":  self.engine_used,
            "alpha":        self.alpha,
            "generated_at": self.generated_at.isoformat(),
            "items": [
                {
                    "item_id":  c.item_id,
                    "s_cf":     c.s_cf,
                    "s_cbf":    c.s_cbf,
                    "s_hybrid": c.s_hybrid,
                    "category": c.category,
                }
                for c in self.items
            ],
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> "RankedList":
        """
        Deserialise from a dict produced by to_cache_dict().

        Called by infrastructure/cache.py on a cache hit.
        Raises KeyError if the dict is malformed (cache corruption).
        """
        items = [
            ScoredCandidate(
                item_id=  int(c["item_id"]),
                s_cf=     float(c["s_cf"]),
                s_cbf=    float(c["s_cbf"]),
                s_hybrid= float(c["s_hybrid"]),
                category= c.get("category", ""),
            )
            for c in data["items"]
        ]
        return cls(
            user_id=      int(data["user_id"]),
            items=        items,
            generated_at= datetime.fromisoformat(data["generated_at"]),
            engine_used=  data.get("engine_used", "hybrid"),
            alpha=        float(data.get("alpha", 0.5)),
        )


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    """
    Runtime configuration for the recommendation pipeline.

    Passed through every stage from candidate generation to ranking.
    Constructed once per request or task invocation and treated as
    immutable throughout the pipeline.

    Attributes
    ----------
    alpha:
        Content-based weight for hybrid fusion.
        S_hybrid = alpha * S_CF + (1 - alpha) * S_CBF.
        1.0 → pure collaborative filtering.
        0.0 → pure content-based filtering.
        Default: 0.5 (equal weight).
    top_n:
        Maximum number of recommendations to return.
        ranking.rank() applies this cutoff after sorting.
        Default: 10.
    diversity_weight:
        MMR diversity penalty weight in [0, 1].
        0.0 → pure relevance ranking (no diversity).
        Higher values penalise consecutive items of the same category.
        Default: 0.0 (disabled).
    candidate_pool_size:
        Maximum number of candidates passed to the scoring engines.
        0 means no cap — use the full unseen-item pool.
        A non-zero value enables the popularity pre-filter in
        candidate_generation.py to keep scoring tractable at scale.
        Default: 0 (uncapped).
    """

    alpha: float = 0.5
    top_n: int = 10
    diversity_weight: float = 0.0
    candidate_pool_size: int = 0

    def __post_init__(self) -> None:
        _validate_unit_float("alpha",            self.alpha)
        _validate_unit_float("diversity_weight", self.diversity_weight)
        _validate_positive_int("top_n",          self.top_n)
        if self.candidate_pool_size < 0:
            raise ValueError(
                f"candidate_pool_size must be >= 0, got {self.candidate_pool_size!r}"
            )

    @property
    def cf_weight(self) -> float:
        """Collaborative filtering weight — complement of alpha."""
        return 1.0 - self.alpha

    @property
    def is_cf_only(self) -> bool:
        """True when alpha == 1.0 (pure collaborative filtering mode)."""
        return self.alpha == 1.0

    @property
    def is_cbf_only(self) -> bool:
        """True when alpha == 0.0 (pure content-based filtering mode)."""
        return self.alpha == 0.0

    @property
    def diversity_enabled(self) -> bool:
        """True when MMR diversity re-ranking is active."""
        return self.diversity_weight > 0.0

    @property
    def pool_is_capped(self) -> bool:
        """True when candidate_pool_size > 0 (popularity pre-filter active)."""
        return self.candidate_pool_size > 0