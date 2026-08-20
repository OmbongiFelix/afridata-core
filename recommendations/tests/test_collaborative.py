"""
Unit tests for CollaborativeEngine.
"""

from unittest.mock import MagicMock
from django.test import TestCase
import numpy as np

from recommendations.domain.engines.collaborative import (
    CollaborativeEngine,
    CF_MODEL_TYPE_SVD,
)
from recommendations.domain.schemas import CandidateSet


class TestCollaborativeEngine(TestCase):
    """Tests for CollaborativeEngine SVD/ALS scoring and cold-start handling."""

    def setUp(self):
        self.engine = CollaborativeEngine()

    def test_unloaded_engine_returns_zero_scores(self):
        candidates = CandidateSet(user_id=1, candidate_ids=[101, 102, 103])
        scores = self.engine.score(user_id=1, candidates=candidates)
        self.assertEqual(scores, {101: 0.0, 102: 0.0, 103: 0.0})

    def test_svd_scoring(self):
        user_factors = np.array([[1.0, 0.5]])
        item_factors = np.array([
            [1.0, 0.0],  # item 101 -> dot product = 1.0
            [0.0, 1.0],  # item 102 -> dot product = 0.5
        ])
        model_dict = {
            "user_factors": user_factors,
            "item_factors": item_factors,
        }

        self.engine._model = model_dict
        self.engine._model_type = CF_MODEL_TYPE_SVD

        scores = self.engine.score_for_user(
            user_id=0,
            candidate_item_ids=[101, 102],
            item_id_to_index={101: 0, 102: 1},
        )

        self.assertIn(101, scores)
        self.assertIn(102, scores)
        self.assertGreater(scores[101], scores[102])

    def test_cold_start_detection(self):
        scores = {101: 0.0, 102: 0.0}
        self.assertTrue(self.engine.is_cold_start(scores))

        scores_active = {101: 0.5, 102: 0.0}
        self.assertFalse(self.engine.is_cold_start(scores_active))
