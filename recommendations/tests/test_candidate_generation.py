"""
Unit tests for the candidate generation engine.

Tests domain/engines/candidate_gen.py in isolation.
All database calls are mocked — no real ORM queries.

Test classes:
  TestCandidateFiltering   — seen items excluded correctly
  TestColdStartCandidate   — no interactions → full pool returned
  TestEmptyPool            — user has seen all items → empty CandidateSet
  TestPopularityPrefilter  — pool capped at candidate_pool_size
  TestCandidateSchema      — returned object is a valid CandidateSet
"""

from django.test import TestCase
from unittest.mock import patch

from recommendations.domain.engines.candidate_generation import CandidateGenerator
from recommendations.domain.schemas import EngineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> EngineConfig:
    """Return an EngineConfig with sensible defaults, accepting overrides."""
    defaults = dict(candidate_pool_size=10)
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _item_ids(n: int, start: int = 1) -> list[int]:
    """Return a list of n consecutive integer item IDs starting at `start`."""
    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# TestCandidateFiltering
# ---------------------------------------------------------------------------

class TestCandidateFiltering(TestCase):
    """Items present in the user's interaction history must be excluded."""

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_seen_items_excluded_from_candidates(
        self, MockInteractionRepo, MockItemRepo
    ):
        """Items the user has already interacted with must not appear in results."""
        all_items = _item_ids(10)          # items 1–10
        seen_items = _item_ids(3)          # items 1–3

        MockItemRepo.return_value.get_popular_items.return_value = all_items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = seen_items

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=42)

        for item_id in seen_items:
            self.assertNotIn(
                item_id,
                result.item_ids,
                msg=f"Seen item {item_id} must not appear in CandidateSet",
            )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_unseen_items_all_present(self, MockInteractionRepo, MockItemRepo):
        """Every item *not* in the user's history must be included."""
        all_items = _item_ids(10)
        seen_items = _item_ids(3)
        expected_unseen = set(all_items) - set(seen_items)

        MockItemRepo.return_value.get_popular_items.return_value = all_items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = seen_items

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=42)

        self.assertEqual(
            set(result.item_ids),
            expected_unseen,
            msg="CandidateSet must contain exactly the unseen items",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_filtering_uses_correct_user_id(self, MockInteractionRepo, MockItemRepo):
        """InteractionRepository must be called with the correct user_id."""
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(5)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config()
        generator = CandidateGenerator(config=config)
        generator.generate(user_id=99)

        MockInteractionRepo.return_value.get_seen_item_ids.assert_called_once_with(
            user_id=99
        )


# ---------------------------------------------------------------------------
# TestColdStartCandidate
# ---------------------------------------------------------------------------

class TestColdStartCandidate(TestCase):
    """A user with no interactions must receive the full candidate pool."""

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_no_interactions_returns_full_pool(self, MockInteractionRepo, MockItemRepo):
        """When seen_items is empty, all pool items should be candidates."""
        all_items = _item_ids(8)

        MockItemRepo.return_value.get_popular_items.return_value = all_items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=1)

        self.assertEqual(
            set(result.item_ids),
            set(all_items),
            msg="Cold-start user should receive every item in the pool",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_no_interactions_no_error(self, MockInteractionRepo, MockItemRepo):
        """generate() must not raise when the user has zero interactions."""
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(5)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config()
        generator = CandidateGenerator(config=config)

        try:
            generator.generate(user_id=7)
        except Exception as exc:  # pragma: no cover
            self.fail(f"generate() raised unexpectedly for a new user: {exc}")


# ---------------------------------------------------------------------------
# TestEmptyPool
# ---------------------------------------------------------------------------

class TestEmptyPool(TestCase):
    """A user who has seen every item must receive an empty CandidateSet."""

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_all_items_seen_returns_empty_candidate_set(
        self, MockInteractionRepo, MockItemRepo
    ):
        """CandidateSet.item_ids must be empty when every item has been seen."""
        all_items = _item_ids(5)

        MockItemRepo.return_value.get_popular_items.return_value = all_items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = all_items

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=3)

        self.assertEqual(
            result.item_ids,
            [],
            msg="CandidateSet.item_ids must be empty when all items are seen",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_all_items_seen_no_exception(self, MockInteractionRepo, MockItemRepo):
        """generate() must not raise when every item has been seen."""
        all_items = _item_ids(5)

        MockItemRepo.return_value.get_popular_items.return_value = all_items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = all_items

        config = _make_config()
        generator = CandidateGenerator(config=config)

        try:
            generator.generate(user_id=3)
        except Exception as exc:  # pragma: no cover
            self.fail(f"generate() raised unexpectedly for a fully-seen pool: {exc}")


# ---------------------------------------------------------------------------
# TestPopularityPrefilter
# ---------------------------------------------------------------------------

class TestPopularityPrefilter(TestCase):
    """The pool must be capped at EngineConfig.candidate_pool_size before filtering."""

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_pool_capped_at_candidate_pool_size(
        self, MockInteractionRepo, MockItemRepo
    ):
        """ItemRepository must be asked for at most candidate_pool_size items."""
        pool_size = 5
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(pool_size)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config(candidate_pool_size=pool_size)
        generator = CandidateGenerator(config=config)
        generator.generate(user_id=10)

        MockItemRepo.return_value.get_popular_items.assert_called_once_with(
            limit=pool_size
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_result_never_exceeds_pool_size(self, MockInteractionRepo, MockItemRepo):
        """Even if no items are filtered, results must not exceed pool_size."""
        pool_size = 4
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(pool_size)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config(candidate_pool_size=pool_size)
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=10)

        self.assertLessEqual(
            len(result.item_ids),
            pool_size,
            msg="CandidateSet must not contain more items than candidate_pool_size",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_different_pool_sizes_respected(self, MockInteractionRepo, MockItemRepo):
        """Varying candidate_pool_size must produce correspondingly sized requests."""
        for pool_size in (3, 7, 15):
            with self.subTest(pool_size=pool_size):
                MockItemRepo.reset_mock()
                MockItemRepo.return_value.get_popular_items.return_value = _item_ids(
                    pool_size
                )
                MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

                config = _make_config(candidate_pool_size=pool_size)
                generator = CandidateGenerator(config=config)
                generator.generate(user_id=10)

                MockItemRepo.return_value.get_popular_items.assert_called_with(
                    limit=pool_size
                )


# ---------------------------------------------------------------------------
# TestCandidateSchema
# ---------------------------------------------------------------------------

class TestCandidateSchema(TestCase):
    """generate() must return a valid CandidateSet dataclass."""

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_returns_candidate_set_instance(self, MockInteractionRepo, MockItemRepo):
        """The return value must be an instance of CandidateSet."""
        from recommendations.domain.schemas import CandidateSet  # local import to mirror spec

        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(5)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config()
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=55)

        self.assertIsInstance(
            result,
            CandidateSet,
            msg="generate() must return a CandidateSet instance",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_user_id_stored_correctly(self, MockInteractionRepo, MockItemRepo):
        """CandidateSet.user_id must match the user_id passed to generate()."""
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(5)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config()
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=55)

        self.assertEqual(
            result.user_id,
            55,
            msg="CandidateSet.user_id must equal the requested user_id",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_item_ids_is_a_list(self, MockInteractionRepo, MockItemRepo):
        """CandidateSet.item_ids must be a list."""
        MockItemRepo.return_value.get_popular_items.return_value = _item_ids(5)
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = []

        config = _make_config()
        generator = CandidateGenerator(config=config)
        result = generator.generate(user_id=55)

        self.assertIsInstance(
            result.item_ids,
            list,
            msg="CandidateSet.item_ids must be a list",
        )

    @patch("recommendations.domain.engines.candidate_gen.ItemRepository")
    @patch("recommendations.domain.engines.candidate_gen.InteractionRepository")
    def test_schema_consistent_across_calls(self, MockInteractionRepo, MockItemRepo):
        """Two generate() calls with identical inputs must return equivalent results."""
        items = _item_ids(6)
        seen = _item_ids(2)

        MockItemRepo.return_value.get_popular_items.return_value = items
        MockInteractionRepo.return_value.get_seen_item_ids.return_value = seen

        config = _make_config()
        generator = CandidateGenerator(config=config)

        result_a = generator.generate(user_id=20)
        result_b = generator.generate(user_id=20)

        self.assertEqual(result_a.user_id, result_b.user_id)
        self.assertEqual(set(result_a.item_ids), set(result_b.item_ids))