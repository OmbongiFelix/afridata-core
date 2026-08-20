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

@dataclass
class CandidateSet:
    """
    Output of the candidate generation stage (Stage 2).

    Carries the pool of dataset IDs that are eligible to be scored for
    a given user. Already filtered to exclude items the user has seen.
    """

    user_id: int
    candidate_ids: List[int] = field(default_factory=list)
    seen_ids: set[int] = field(default_factory=set)
    is_cold_start: bool = False
    total_pool_size: int = 0
    item_id_to_index: dict[int, int] = field(default_factory=dict)

    def __init__(
        self,
        user_id: int,
        candidate_ids: List[int] | None = None,
        item_ids: List[int] | None = None,
        seen_ids: set[int] | None = None,
        is_cold_start: bool = False,
        total_pool_size: int = 0,
        item_id_to_index: dict[int, int] | None = None,
    ) -> None:
        self.user_id = int(user_id)
        raw_ids = candidate_ids if candidate_ids is not None else (item_ids or [])
        self.candidate_ids = [int(x) for x in raw_ids]
        self.seen_ids = set(seen_ids) if seen_ids is not None else set()
        self.is_cold_start = is_cold_start
        self.total_pool_size = total_pool_size or len(self.candidate_ids)
        if item_id_to_index is not None:
            self.item_id_to_index = item_id_to_index
        else:
            self.item_id_to_index = {iid: idx for idx, iid in enumerate(self.candidate_ids)}

    @property
    def item_ids(self) -> List[int]:
        """Alias for candidate_ids."""
        return self.candidate_ids

    @property
    def is_empty(self) -> bool:
        """True when there are no eligible candidates."""
        return len(self.candidate_ids) == 0

    @property
    def size(self) -> int:
        """Number of candidate items in the pool."""
        return len(self.candidate_ids)

    def __len__(self) -> int:
        return len(self.candidate_ids)

    def __iter__(self):
        return iter(self.candidate_ids)


# ---------------------------------------------------------------------------
# ScoredCandidate
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """
    A single dataset item with component and fused scores.
    """

    item_id: Any
    s_cf: float = 0.0
    s_cbf: float = 0.0
    s_hybrid: float = 0.0
    category: str = ""
    score: float = 0.0

    def __init__(
        self,
        item_id: Any,
        s_cf: float = 0.0,
        s_cbf: float = 0.0,
        s_hybrid: float = 0.0,
        category: str = "",
        score: float | None = None,
    ) -> None:
        self.item_id = item_id
        self.s_cf = float(s_cf)
        self.s_cbf = float(s_cbf)
        fused = float(score) if score is not None else float(s_hybrid)
        self.s_hybrid = fused
        self.score = fused
        self.category = str(category or "")

    @property
    def is_cold_start(self) -> bool:
        """True when both component scores are zero (fully cold-start user)."""
        return self.s_cf == 0.0 and self.s_cbf == 0.0

    def __repr__(self) -> str:
        cat = f", category={self.category!r}" if self.category else ""
        return (
            f"ScoredCandidate(item_id={self.item_id}, s_hybrid={self.s_hybrid:.4f}"
            f", s_cf={self.s_cf:.4f}, s_cbf={self.s_cbf:.4f}{cat})"
        )


# ---------------------------------------------------------------------------
# RankedList
# ---------------------------------------------------------------------------

@dataclass
class RankedList:
    """
    Ordered Top-N recommendation list for a user.
    """

    user_id: int = 0
    items: List[ScoredCandidate] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    engine_used: str = "hybrid"
    alpha: float = 0.5

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @property
    def top_n(self) -> int:
        return len(self.items)

    @property
    def ranked_dataset_ids(self) -> List[Any]:
        return [item.item_id for item in self.items]

    @property
    def scores(self) -> dict[Any, float]:
        """Dictionary of item_id -> score for compatibility with tests."""
        return {item.item_id: item.s_hybrid for item in self.items}

    def to_cache_dict(self) -> dict:
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
        items = [
            ScoredCandidate(
                item_id=  c["item_id"],
                s_cf=     float(c.get("s_cf", 0.0)),
                s_cbf=    float(c.get("s_cbf", 0.0)),
                s_hybrid= float(c.get("s_hybrid", 0.0)),
                category= c.get("category", ""),
            )
            for c in data["items"]
        ]
        return cls(
            user_id=      int(data.get("user_id", 0)),
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
    auto_cold_start: bool = True

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