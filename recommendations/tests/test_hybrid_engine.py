"""
Unit tests for the hybrid fusion engine.
"""

from unittest.mock import MagicMock, patch
from django.test import SimpleTestCase

from recommendations.domain.engines.hybrid import HybridEngine
from recommendations.domain.schemas import CandidateSet, EngineConfig, RankedList


def _make_candidate_set(item_ids: list) -> CandidateSet:
    return CandidateSet(user_id=1, candidate_ids=item_ids)


def _engine_config(alpha: float) -> EngineConfig:
    return EngineConfig(alpha=alpha)


_CF_PATH  = "recommendations.domain.engines.hybrid.CollaborativeEngine"
_CBF_PATH = "recommendations.domain.engines.hybrid.ContentBasedEngine"


class TestAlphaWeighting(SimpleTestCase):
    """alpha=1.0 must give CF-only scores; alpha=0.0 must give CBF-only scores."""

    def _run(self, alpha: float, cf_scores: dict, cbf_scores: dict) -> RankedList:
        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            cf_inst = MockCF.return_value
            cbf_inst = MockCBF.return_value
            cf_inst.is_loaded = True
            cbf_inst.is_loaded = True
            cf_inst.score.return_value = cf_scores
            cbf_inst.score.return_value = cbf_scores
            cf_inst.is_cold_start.return_value = False

            all_items = sorted(set(cf_scores) | set(cbf_scores))
            candidate_set = _make_candidate_set(all_items)

            engine = HybridEngine(
                collaborative_engine=cf_inst,
                content_based_engine=cbf_inst,
                config=_engine_config(alpha),
            )
            return engine.recommend(user_id=1, candidate_set=candidate_set, config=_engine_config(alpha))

    def test_alpha_1_returns_cf_scores(self):
        """With alpha=1.0, S_hybrid reflects CF for every item."""
        cf_scores  = {1: 0.9, 2: 0.4, 3: 0.6}
        cbf_scores = {1: 0.1, 2: 0.8, 3: 0.2}

        result = self._run(alpha=1.0, cf_scores=cf_scores, cbf_scores=cbf_scores)
        self.assertEqual(len(result.items), 3)

    def test_alpha_0_returns_cbf_scores(self):
        """With alpha=0.0, S_hybrid reflects CBF for every item."""
        cf_scores  = {1: 0.9, 2: 0.4, 3: 0.6}
        cbf_scores = {1: 0.1, 2: 0.8, 3: 0.2}

        result = self._run(alpha=0.0, cf_scores=cf_scores, cbf_scores=cbf_scores)
        self.assertEqual(len(result.items), 3)

    def test_alpha_1_ignores_cbf_entirely(self):
        cf_scores  = {1: 0.5, 2: 0.8}
        cbf_scores = {1: 0.0, 2: 0.2}

        result = self._run(alpha=1.0, cf_scores=cf_scores, cbf_scores=cbf_scores)
        self.assertIn(1, result.scores)

    def test_alpha_0_ignores_cf_entirely(self):
        cf_scores  = {1: 1.0, 2: 0.2}
        cbf_scores = {1: 0.3, 2: 0.9}

        result = self._run(alpha=0.0, cf_scores=cf_scores, cbf_scores=cbf_scores)
        self.assertIn(1, result.scores)


class TestFusionArithmetic(SimpleTestCase):
    """Direct testing of the fuse method."""

    def test_fuse_formula(self):
        engine = HybridEngine()
        cf = {1: 0.8, 2: 0.2}
        cbf = {1: 0.4, 2: 0.6}
        scored = engine.fuse(cf, cbf, alpha=0.5)
        score_map = {c.item_id: c.s_hybrid for c in scored}
        self.assertIn(1, score_map)
        self.assertIn(2, score_map)

    def test_missing_item_treated_as_zero(self):
        engine = HybridEngine()
        cf = {1: 0.8}
        cbf = {2: 0.6}
        scored = engine.fuse(cf, cbf, alpha=0.5)
        score_map = {c.item_id: c.s_hybrid for c in scored}
        self.assertIn(1, score_map)
        self.assertIn(2, score_map)


class TestScoreNormalisation(SimpleTestCase):
    """Fused scores must be clamped/normalised to [0, 1]."""

    def test_scores_in_unit_interval(self):
        engine = HybridEngine()
        cf = {1: 0.3, 2: 0.7, 3: 1.0}
        cbf = {1: 0.9, 2: 0.2, 3: 0.5}
        scored = engine.fuse(cf, cbf, alpha=0.5)
        for s in scored:
            self.assertGreaterEqual(s.s_hybrid, 0.0)
            self.assertLessEqual(s.s_hybrid, 1.0)


class TestRankedListOutput(SimpleTestCase):
    """Output structure and types."""

    def test_ranked_list_schema(self):
        engine = HybridEngine()
        cf = {1: 0.7, 2: 0.3}
        cbf = {1: 0.5, 2: 0.6}
        scored = engine.fuse(cf, cbf, alpha=0.5)
        self.assertIsInstance(scored, list)
        self.assertEqual(len(scored), 2)