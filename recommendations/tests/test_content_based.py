"""
Unit tests for ContentBasedEngine.
"""

from unittest.mock import patch, MagicMock
from django.test import TestCase
import numpy as np
from scipy import sparse

from recommendations.domain.engines.content_based import ContentBasedEngine
from recommendations.domain.schemas import CandidateSet


class TestContentBasedEngine(TestCase):
    """Tests for ContentBasedEngine TF-IDF scoring and popularity fallback."""

    def setUp(self):
        self.engine = ContentBasedEngine()

    def test_popularity_fallback_on_cold_start(self):
        candidates = CandidateSet(user_id=1, candidate_ids=[101, 102, 103])
        popularities = {101: 50.0, 102: 100.0, 103: 10.0}

        scores = self.engine._popularity_scores(
            candidate_item_ids=[101, 102, 103],
            item_popularities=popularities,
            exclude_set=set(),
        )

        self.assertEqual(scores[102], 1.0)
        self.assertEqual(scores[103], 0.1)
        self.assertAlmostEqual(scores[101], 0.5, places=2)

    def test_score_with_loaded_matrix(self):
        # 3 items with 2 features
        matrix = sparse.csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ])
        item_ids = [101, 102, 103]

        self.engine._tfidf_matrix = matrix
        self.engine._item_ids = item_ids

        # User interacted with item 101
        scores = self.engine.score_for_user(
            interacted_item_ids=[101],
            interaction_weights=[1.0],
            candidate_item_ids=[102, 103],
            item_popularities={},
            exclude_interacted=True,
        )

        self.assertIn(102, scores)
        self.assertIn(103, scores)
        # 103 has cosine similarity with 101 of 0.8, 102 has 0.0
        self.assertGreater(scores[103], scores[102])

    def test_unloaded_engine_falls_back_gracefully(self):
        candidates = CandidateSet(user_id=1, candidate_ids=[101, 102])
        scores = self.engine.score(
            user_id=1,
            candidates=candidates,
            item_popularities={101: 10, 102: 20},
        )
        self.assertIn(101, scores)
        self.assertIn(102, scores)
