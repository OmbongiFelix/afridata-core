"""
Unit tests for the candidate generation engine.
"""

from unittest.mock import patch
from django.test import TestCase

from recommendations.domain.engines.candidate_generation import CandidateGenerator
from recommendations.domain.schemas import CandidateSet, EngineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> EngineConfig:
    defaults = dict(candidate_pool_size=10)
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _item_ids(n: int, start: int = 1) -> list[int]:
    return list(range(start, start + n))


def _make_interactions(item_ids: list[int]) -> list[dict]:
    return [{"dataset_id": iid, "weight": 1.0} for iid in item_ids]


# ---------------------------------------------------------------------------
# TestCandidateFiltering
# ---------------------------------------------------------------------------

class TestCandidateFiltering(TestCase):
    """Items present in the user's interaction history must be excluded."""

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_seen_items_excluded_from_candidates(
        self, mock_interactions, mock_datasets
    ):
        all_items = _item_ids(10)
        seen_items = _item_ids(3)

        mock_datasets.return_value = all_items
        mock_interactions.return_value = _make_interactions(seen_items)

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator()
        result = generator.generate(user_id=42, config=config)

        for item_id in seen_items:
            self.assertNotIn(
                item_id,
                result.item_ids,
                msg=f"Seen item {item_id} must not appear in CandidateSet",
            )

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_unseen_items_all_present(self, mock_interactions, mock_datasets):
        all_items = _item_ids(10)
        seen_items = _item_ids(3)
        expected_unseen = set(all_items) - set(seen_items)

        mock_datasets.return_value = all_items
        mock_interactions.return_value = _make_interactions(seen_items)

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator()
        result = generator.generate(user_id=42, config=config)

        self.assertEqual(
            set(result.item_ids),
            expected_unseen,
            msg="CandidateSet must contain exactly the unseen items",
        )

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_filtering_uses_correct_user_id(self, mock_interactions, mock_datasets):
        mock_datasets.return_value = _item_ids(5)
        mock_interactions.return_value = []

        config = _make_config()
        generator = CandidateGenerator()
        generator.generate(user_id=99, config=config)

        mock_interactions.assert_called_once_with(99)


# ---------------------------------------------------------------------------
# TestColdStartCandidate
# ---------------------------------------------------------------------------

class TestColdStartCandidate(TestCase):
    """A user with no interactions must receive the full candidate pool."""

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_no_interactions_returns_full_pool(self, mock_interactions, mock_datasets):
        all_items = _item_ids(8)

        mock_datasets.return_value = all_items
        mock_interactions.return_value = []

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator()
        result = generator.generate(user_id=1, config=config)

        self.assertEqual(
            set(result.item_ids),
            set(all_items),
            msg="Cold-start user should receive every item in the pool",
        )
        self.assertTrue(result.is_cold_start)

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_no_interactions_no_error(self, mock_interactions, mock_datasets):
        mock_datasets.return_value = _item_ids(5)
        mock_interactions.return_value = []

        config = _make_config()
        generator = CandidateGenerator()

        try:
            generator.generate(user_id=7, config=config)
        except Exception as exc:
            self.fail(f"generate() raised unexpectedly for a new user: {exc}")


# ---------------------------------------------------------------------------
# TestEmptyPool
# ---------------------------------------------------------------------------

class TestEmptyPool(TestCase):
    """A user who has seen every item must receive an empty CandidateSet."""

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_all_items_seen_returns_empty_candidate_set(
        self, mock_interactions, mock_datasets
    ):
        all_items = _item_ids(5)

        mock_datasets.return_value = all_items
        mock_interactions.return_value = _make_interactions(all_items)

        config = _make_config(candidate_pool_size=20)
        generator = CandidateGenerator()
        result = generator.generate(user_id=3, config=config)

        self.assertEqual(
            result.item_ids,
            [],
            msg="CandidateSet.item_ids must be empty when all items are seen",
        )
        self.assertTrue(result.is_empty)


# ---------------------------------------------------------------------------
# TestCandidateSchema
# ---------------------------------------------------------------------------

class TestCandidateSchema(TestCase):
    """generate() must return a valid CandidateSet dataclass."""

    @patch("recommendations.domain.engines.candidate_generation.get_all_dataset_ids")
    @patch("recommendations.domain.engines.candidate_generation.get_user_interactions")
    def test_returns_candidate_set_instance(self, mock_interactions, mock_datasets):
        mock_datasets.return_value = _item_ids(5)
        mock_interactions.return_value = []

        config = _make_config()
        generator = CandidateGenerator()
        result = generator.generate(user_id=55, config=config)

        self.assertIsInstance(
            result,
            CandidateSet,
            msg="generate() must return a CandidateSet instance",
        )
        self.assertEqual(result.user_id, 55)
        self.assertIsInstance(result.item_ids, list)