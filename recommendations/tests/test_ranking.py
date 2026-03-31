"""
Unit tests for the ranking module.

Tests domain/ranking.py in isolation.
No database, no cache, no engine calls — purely functional.

Test classes:
  TestScoreOrdering   — candidates sorted by s_hybrid descending
  TestTopNCutoff      — exactly N items returned when pool > N
  TestTieBreaking     — ties resolved by item_id ascending (deterministic)
  TestEmptyInput      — empty list returns empty RankedList without error
  TestDiversityRerank — MMR re-ranking reduces same-category clustering
  TestRankedListSchema — returned object is a valid RankedList dataclass
"""

from django.test import SimpleTestCase

from recommendations.domain.ranking import rank
from recommendations.domain.schemas import ScoredCandidate, EngineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(item_id: str, s_hybrid: float, category: str = "default") -> ScoredCandidate:
    """Convenience factory — only the fields ranking cares about are required."""
    return ScoredCandidate(item_id=item_id, s_hybrid=s_hybrid, category=category)


def _default_config(**overrides) -> EngineConfig:
    """Return a minimal EngineConfig, allowing targeted field overrides."""
    defaults = dict(top_n=10, diversity_weight=0.0)
    defaults.update(overrides)
    return EngineConfig(**defaults)


# ---------------------------------------------------------------------------
# TestScoreOrdering
# ---------------------------------------------------------------------------

class TestScoreOrdering(SimpleTestCase):
    """rank() must return candidates sorted by s_hybrid descending."""

    def test_descending_order_basic(self):
        candidates = [
            _make_candidate("a", 0.3),
            _make_candidate("b", 0.9),
            _make_candidate("c", 0.6),
        ]
        config = _default_config(top_n=3)
        result = rank(candidates, config)
        scores = [item.s_hybrid for item in result.items]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_descending_order_single_item(self):
        candidates = [_make_candidate("only", 0.5)]
        config = _default_config(top_n=5)
        result = rank(candidates, config)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].item_id, "only")

    def test_descending_order_already_sorted(self):
        candidates = [
            _make_candidate("x", 1.0),
            _make_candidate("y", 0.7),
            _make_candidate("z", 0.2),
        ]
        config = _default_config(top_n=3)
        result = rank(candidates, config)
        self.assertEqual(
            [item.item_id for item in result.items], ["x", "y", "z"]
        )

    def test_descending_order_reverse_input(self):
        candidates = [
            _make_candidate("low",  0.1),
            _make_candidate("mid",  0.5),
            _make_candidate("high", 0.9),
        ]
        config = _default_config(top_n=3)
        result = rank(candidates, config)
        self.assertEqual(result.items[0].item_id, "high")
        self.assertEqual(result.items[-1].item_id, "low")


# ---------------------------------------------------------------------------
# TestTopNCutoff
# ---------------------------------------------------------------------------

class TestTopNCutoff(SimpleTestCase):
    """rank() returns exactly config.top_n items when pool is larger."""

    def test_exact_cutoff(self):
        candidates = [_make_candidate(str(i), float(i)) for i in range(20)]
        config = _default_config(top_n=5)
        result = rank(candidates, config)
        self.assertEqual(len(result.items), 5)

    def test_top_n_items_are_highest_scoring(self):
        candidates = [_make_candidate(str(i), float(i)) for i in range(10)]
        config = _default_config(top_n=3)
        result = rank(candidates, config)
        # Best 3 are scores 9, 8, 7
        returned_ids = {item.item_id for item in result.items}
        self.assertEqual(returned_ids, {"9", "8", "7"})

    def test_pool_equal_to_top_n(self):
        candidates = [_make_candidate(str(i), float(i)) for i in range(5)]
        config = _default_config(top_n=5)
        result = rank(candidates, config)
        self.assertEqual(len(result.items), 5)

    def test_pool_smaller_than_top_n(self):
        """When fewer candidates exist than top_n, return all of them."""
        candidates = [_make_candidate(str(i), float(i)) for i in range(3)]
        config = _default_config(top_n=10)
        result = rank(candidates, config)
        self.assertEqual(len(result.items), 3)


# ---------------------------------------------------------------------------
# TestTieBreaking
# ---------------------------------------------------------------------------

class TestTieBreaking(SimpleTestCase):
    """Ties in s_hybrid must be broken by item_id ascending (deterministic)."""

    def test_tie_broken_by_item_id_ascending(self):
        candidates = [
            _make_candidate("c_item", 0.5),
            _make_candidate("a_item", 0.5),
            _make_candidate("b_item", 0.5),
        ]
        config = _default_config(top_n=3)
        result = rank(candidates, config)
        ids = [item.item_id for item in result.items]
        self.assertEqual(ids, ["a_item", "b_item", "c_item"])

    def test_partial_tie(self):
        """Only tied items are reordered by item_id; higher scores stay on top."""
        candidates = [
            _make_candidate("z_item", 0.9),   # clear winner
            _make_candidate("c_item", 0.5),
            _make_candidate("a_item", 0.5),
            _make_candidate("b_item", 0.5),
        ]
        config = _default_config(top_n=4)
        result = rank(candidates, config)
        ids = [item.item_id for item in result.items]
        self.assertEqual(ids[0], "z_item")
        self.assertEqual(ids[1:], ["a_item", "b_item", "c_item"])

    def test_tie_breaking_is_stable_across_calls(self):
        """Calling rank() twice with the same input produces identical output."""
        candidates = [
            _make_candidate("m_item", 0.5),
            _make_candidate("a_item", 0.5),
            _make_candidate("z_item", 0.5),
        ]
        config = _default_config(top_n=3)
        result_a = rank(candidates, config)
        result_b = rank(candidates, config)
        self.assertEqual(
            [i.item_id for i in result_a.items],
            [i.item_id for i in result_b.items],
        )


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------

class TestEmptyInput(SimpleTestCase):
    """Empty candidate list must return an empty RankedList — no exception."""

    def test_empty_list_returns_empty_ranked_list(self):
        config = _default_config(top_n=10)
        result = rank([], config)
        self.assertEqual(result.items, [])

    def test_empty_list_does_not_raise(self):
        config = _default_config(top_n=10)
        try:
            rank([], config)
        except Exception as exc:  # pragma: no cover
            self.fail(f"rank() raised an unexpected exception on empty input: {exc}")

    def test_empty_list_result_has_timestamp(self):
        """Even an empty result must carry a generated_at timestamp."""
        config = _default_config(top_n=10)
        result = rank([], config)
        self.assertIsNotNone(result.generated_at)


# ---------------------------------------------------------------------------
# TestDiversityRerank
# ---------------------------------------------------------------------------

class TestDiversityRerank(SimpleTestCase):
    """MMR re-ranking must reduce same-category consecutive placements."""

    def _count_consecutive_same_category(self, items) -> int:
        """Count adjacent pairs that share the same category."""
        return sum(
            1
            for a, b in zip(items, items[1:])
            if a.category == b.category
        )

    def test_diversity_reduces_consecutive_same_category(self):
        # 6 items in category "A", 2 in category "B"
        candidates = [
            _make_candidate("a1", 0.95, category="A"),
            _make_candidate("a2", 0.90, category="A"),
            _make_candidate("a3", 0.85, category="A"),
            _make_candidate("a4", 0.80, category="A"),
            _make_candidate("a5", 0.75, category="A"),
            _make_candidate("a6", 0.70, category="A"),
            _make_candidate("b1", 0.65, category="B"),
            _make_candidate("b2", 0.60, category="B"),
        ]
        config_no_diversity  = _default_config(top_n=8, diversity_weight=0.0)
        config_with_diversity = _default_config(top_n=8, diversity_weight=0.5)

        result_no_div   = rank(candidates, config_no_diversity)
        result_with_div = rank(candidates, config_with_diversity)

        consecutive_no_div   = self._count_consecutive_same_category(result_no_div.items)
        consecutive_with_div = self._count_consecutive_same_category(result_with_div.items)

        self.assertLess(
            consecutive_with_div,
            consecutive_no_div,
            msg=(
                "MMR re-ranking should produce fewer consecutive same-category items "
                f"(got {consecutive_with_div} vs {consecutive_no_div} without diversity)."
            ),
        )

    def test_zero_diversity_weight_preserves_score_order(self):
        candidates = [
            _make_candidate("a1", 0.9, category="A"),
            _make_candidate("a2", 0.8, category="A"),
            _make_candidate("b1", 0.7, category="B"),
        ]
        config = _default_config(top_n=3, diversity_weight=0.0)
        result = rank(candidates, config)
        scores = [item.s_hybrid for item in result.items]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_all_same_category_does_not_raise(self):
        """Full diversity weight with a single category must not raise."""
        candidates = [_make_candidate(str(i), float(i) / 10, category="X") for i in range(5)]
        config = _default_config(top_n=5, diversity_weight=1.0)
        try:
            rank(candidates, config)
        except Exception as exc:  # pragma: no cover
            self.fail(f"rank() raised with all-same-category input: {exc}")


# ---------------------------------------------------------------------------
# TestRankedListSchema
# ---------------------------------------------------------------------------

class TestRankedListSchema(SimpleTestCase):
    """The returned object must be a valid RankedList dataclass."""

    def _rank_small(self):
        candidates = [
            _make_candidate("item1", 0.8),
            _make_candidate("item2", 0.5),
            _make_candidate("item3", 0.3),
        ]
        config = _default_config(top_n=3)
        return rank(candidates, config)

    def test_result_has_items_attribute(self):
        result = self._rank_small()
        self.assertTrue(hasattr(result, "items"))

    def test_result_items_are_scored_candidates(self):
        result = self._rank_small()
        for item in result.items:
            self.assertIsInstance(item, ScoredCandidate)

    def test_result_has_non_none_generated_at(self):
        result = self._rank_small()
        self.assertIsNotNone(result.generated_at)

    def test_result_generated_at_is_consistent_type(self):
        """generated_at should be a datetime (or at least not a bare string)."""
        import datetime
        result = self._rank_small()
        self.assertIsInstance(result.generated_at, datetime.datetime)