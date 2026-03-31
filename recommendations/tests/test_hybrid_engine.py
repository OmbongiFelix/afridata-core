"""
Unit tests for the hybrid fusion engine.

Tests domain/engines/hybrid.py in isolation.
Both CollaborativeEngine and ContentBasedEngine are mocked.
This file tests fusion logic only — not engine correctness.

Test classes:
  TestAlphaWeighting    — alpha=1.0 gives CF-only; alpha=0.0 gives CBF-only
  TestFusionArithmetic  — alpha=0.5 gives arithmetic mean of both scores
  TestMissingItemScore  — item absent from one engine treated as 0.0
  TestScoreNormalisation — fused scores clamped to [0, 1]
  TestRankedListOutput  — return type is a valid RankedList from ranking.py
"""

from django.test import SimpleTestCase
from unittest.mock import MagicMock, patch
from recommendations.domain.engines.hybrid import HybridEngine
from recommendations.domain.schemas import CandidateSet, EngineConfig, RankedList


def _make_candidate_set(scores: dict) -> CandidateSet:
    """Helper: build a CandidateSet from a plain {item_id: score} dict."""
    cs = MagicMock(spec=CandidateSet)
    cs.scores = scores
    cs.__iter__ = lambda self: iter(scores.items())
    return cs


def _make_ranked_list(scores: dict) -> RankedList:
    """Helper: build a mock RankedList from a {item_id: score} dict."""
    rl = MagicMock(spec=RankedList)
    rl.scores = scores
    rl.items = sorted(scores, key=scores.__getitem__, reverse=True)
    return rl


def _engine_config(alpha: float) -> EngineConfig:
    """Helper: build a minimal EngineConfig with the given alpha."""
    cfg = MagicMock(spec=EngineConfig)
    cfg.alpha = alpha
    return cfg


# ---------------------------------------------------------------------------
# Shared mock targets
# ---------------------------------------------------------------------------
_CF_PATH  = "recommendations.domain.engines.hybrid.CollaborativeEngine"
_CBF_PATH = "recommendations.domain.engines.hybrid.ContentBasedEngine"


class TestAlphaWeighting(SimpleTestCase):
    """alpha=1.0 must give CF-only scores; alpha=0.0 must give CBF-only scores."""

    def _run(self, alpha: float, cf_scores: dict, cbf_scores: dict) -> RankedList:
        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            MockCF.return_value.recommend.return_value  = _make_candidate_set(cf_scores)
            MockCBF.return_value.recommend.return_value = _make_candidate_set(cbf_scores)

            engine = HybridEngine(config=_engine_config(alpha))
            return engine.recommend(user_id=1, candidate_set=MagicMock(spec=CandidateSet))

    def test_alpha_1_returns_cf_scores(self):
        """With alpha=1.0, S_hybrid == S_CF for every item."""
        cf_scores  = {"item_a": 0.9, "item_b": 0.4, "item_c": 0.6}
        cbf_scores = {"item_a": 0.1, "item_b": 0.8, "item_c": 0.2}

        result = self._run(alpha=1.0, cf_scores=cf_scores, cbf_scores=cbf_scores)

        for item_id, cf_score in cf_scores.items():
            self.assertAlmostEqual(
                result.scores[item_id], cf_score, places=6,
                msg=f"alpha=1.0: expected S_hybrid[{item_id}] == S_CF={cf_score}"
            )

    def test_alpha_0_returns_cbf_scores(self):
        """With alpha=0.0, S_hybrid == S_CBF for every item."""
        cf_scores  = {"item_a": 0.9, "item_b": 0.4, "item_c": 0.6}
        cbf_scores = {"item_a": 0.1, "item_b": 0.8, "item_c": 0.2}

        result = self._run(alpha=0.0, cf_scores=cf_scores, cbf_scores=cbf_scores)

        for item_id, cbf_score in cbf_scores.items():
            self.assertAlmostEqual(
                result.scores[item_id], cbf_score, places=6,
                msg=f"alpha=0.0: expected S_hybrid[{item_id}] == S_CBF={cbf_score}"
            )

    def test_alpha_1_ignores_cbf_entirely(self):
        """With alpha=1.0, CBF scores have no influence even if very different."""
        cf_scores  = {"item_x": 0.5}
        cbf_scores = {"item_x": 0.0}   # extreme difference — should be ignored

        result = self._run(alpha=1.0, cf_scores=cf_scores, cbf_scores=cbf_scores)

        self.assertAlmostEqual(result.scores["item_x"], 0.5, places=6)

    def test_alpha_0_ignores_cf_entirely(self):
        """With alpha=0.0, CF scores have no influence even if very different."""
        cf_scores  = {"item_x": 1.0}   # extreme difference — should be ignored
        cbf_scores = {"item_x": 0.3}

        result = self._run(alpha=0.0, cf_scores=cf_scores, cbf_scores=cbf_scores)

        self.assertAlmostEqual(result.scores["item_x"], 0.3, places=6)


class TestFusionArithmetic(SimpleTestCase):
    """alpha=0.5 must produce the arithmetic mean of S_CF and S_CBF."""

    def _run_fusion(self, cf_scores: dict, cbf_scores: dict) -> RankedList:
        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            MockCF.return_value.recommend.return_value  = _make_candidate_set(cf_scores)
            MockCBF.return_value.recommend.return_value = _make_candidate_set(cbf_scores)

            engine = HybridEngine(config=_engine_config(alpha=0.5))
            return engine.recommend(user_id=1, candidate_set=MagicMock(spec=CandidateSet))

    def test_mean_of_equal_scores(self):
        """Mean of identical scores equals that score."""
        result = self._run_fusion(
            cf_scores={"item_a": 0.6},
            cbf_scores={"item_a": 0.6},
        )
        self.assertAlmostEqual(result.scores["item_a"], 0.6, places=6)

    def test_mean_of_differing_scores(self):
        """Mean of 0.8 and 0.4 must be 0.6."""
        result = self._run_fusion(
            cf_scores={"item_a": 0.8},
            cbf_scores={"item_a": 0.4},
        )
        self.assertAlmostEqual(result.scores["item_a"], 0.6, places=6)

    def test_mean_across_multiple_items(self):
        """Arithmetic mean is applied independently to each item."""
        cf_scores  = {"item_a": 1.0, "item_b": 0.0, "item_c": 0.7}
        cbf_scores = {"item_a": 0.0, "item_b": 1.0, "item_c": 0.3}
        expected   = {"item_a": 0.5, "item_b": 0.5, "item_c": 0.5}

        result = self._run_fusion(cf_scores=cf_scores, cbf_scores=cbf_scores)

        for item_id, exp_score in expected.items():
            self.assertAlmostEqual(
                result.scores[item_id], exp_score, places=6,
                msg=f"alpha=0.5: expected mean for {item_id} == {exp_score}"
            )

    def test_formula_is_weighted_not_max(self):
        """Verify fusion uses the weighted sum, not max or min."""
        result = self._run_fusion(
            cf_scores={"item_a": 0.9},
            cbf_scores={"item_a": 0.1},
        )
        # max would be 0.9, min would be 0.1 — only mean is 0.5
        self.assertAlmostEqual(result.scores["item_a"], 0.5, places=6)


class TestMissingItemScore(SimpleTestCase):
    """Items absent from one engine's output must be treated as score=0.0."""

    def _run_with_gap(self, alpha: float, cf_scores: dict, cbf_scores: dict) -> RankedList:
        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            MockCF.return_value.recommend.return_value  = _make_candidate_set(cf_scores)
            MockCBF.return_value.recommend.return_value = _make_candidate_set(cbf_scores)

            engine = HybridEngine(config=_engine_config(alpha=alpha))
            return engine.recommend(user_id=1, candidate_set=MagicMock(spec=CandidateSet))

    def test_item_missing_from_cbf_uses_zero(self):
        """Item in CF but not CBF: S_CBF is treated as 0.0, not skipped."""
        result = self._run_with_gap(
            alpha=0.5,
            cf_scores={"item_only_in_cf": 0.8},
            cbf_scores={},
        )
        # 0.5 * 0.8 + 0.5 * 0.0 = 0.4
        self.assertIn("item_only_in_cf", result.scores)
        self.assertAlmostEqual(result.scores["item_only_in_cf"], 0.4, places=6)

    def test_item_missing_from_cf_uses_zero(self):
        """Item in CBF but not CF: S_CF is treated as 0.0, not skipped."""
        result = self._run_with_gap(
            alpha=0.5,
            cf_scores={},
            cbf_scores={"item_only_in_cbf": 0.6},
        )
        # 0.5 * 0.0 + 0.5 * 0.6 = 0.3
        self.assertIn("item_only_in_cbf", result.scores)
        self.assertAlmostEqual(result.scores["item_only_in_cbf"], 0.3, places=6)

    def test_missing_item_not_dropped_from_output(self):
        """Missing item must appear in ranked output, not be silently dropped."""
        result = self._run_with_gap(
            alpha=0.8,
            cf_scores={"present": 0.9, "cf_only": 0.5},
            cbf_scores={"present": 0.7},
        )
        self.assertIn("cf_only", result.scores,
                      "Item present only in CF must not be silently dropped")

    def test_alpha_1_missing_cbf_still_uses_cf_score(self):
        """With alpha=1.0, a missing CBF score should not zero out the CF score."""
        result = self._run_with_gap(
            alpha=1.0,
            cf_scores={"item_a": 0.9},
            cbf_scores={},
        )
        # 1.0 * 0.9 + 0.0 * 0.0 = 0.9
        self.assertAlmostEqual(result.scores["item_a"], 0.9, places=6)


class TestScoreNormalisation(SimpleTestCase):
    """Fused scores must be clamped/normalised to [0, 1] before ranking."""

    def _run_normalised(self, cf_scores: dict, cbf_scores: dict,
                        alpha: float = 0.5) -> RankedList:
        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            MockCF.return_value.recommend.return_value  = _make_candidate_set(cf_scores)
            MockCBF.return_value.recommend.return_value = _make_candidate_set(cbf_scores)

            engine = HybridEngine(config=_engine_config(alpha=alpha))
            return engine.recommend(user_id=1, candidate_set=MagicMock(spec=CandidateSet))

    def test_all_scores_within_unit_interval(self):
        """Every score in the result must satisfy 0.0 <= score <= 1.0."""
        result = self._run_normalised(
            cf_scores={"a": 0.3, "b": 0.7, "c": 1.0},
            cbf_scores={"a": 0.9, "b": 0.2, "c": 0.5},
        )
        for item_id, score in result.scores.items():
            self.assertGreaterEqual(score, 0.0,
                                    f"Score for {item_id} is below 0.0: {score}")
            self.assertLessEqual(score, 1.0,
                                 f"Score for {item_id} is above 1.0: {score}")

    def test_score_does_not_exceed_one(self):
        """Boundary: max possible input (both engines score 1.0) must not exceed 1.0."""
        result = self._run_normalised(
            cf_scores={"item_max": 1.0},
            cbf_scores={"item_max": 1.0},
        )
        self.assertLessEqual(result.scores["item_max"], 1.0)

    def test_score_not_below_zero(self):
        """Boundary: min possible input (both engines score 0.0) must not go below 0.0."""
        result = self._run_normalised(
            cf_scores={"item_min": 0.0},
            cbf_scores={"item_min": 0.0},
        )
        self.assertGreaterEqual(result.scores["item_min"], 0.0)

    def test_normalisation_preserves_relative_order(self):
        """Normalisation must not invert the ranking of items."""
        cf_scores  = {"high": 0.9, "low": 0.1}
        cbf_scores = {"high": 0.8, "low": 0.2}

        result = self._run_normalised(cf_scores=cf_scores, cbf_scores=cbf_scores)

        self.assertGreater(
            result.scores["high"], result.scores["low"],
            "Normalisation must preserve relative ordering of scores"
        )


class TestRankedListOutput(SimpleTestCase):
    """The return value must be a RankedList dataclass instance."""

    def _call_engine(self, alpha: float = 0.5) -> object:
        cf_scores  = {"item_a": 0.7, "item_b": 0.3}
        cbf_scores = {"item_a": 0.5, "item_b": 0.6}

        with patch(_CF_PATH) as MockCF, patch(_CBF_PATH) as MockCBF:
            MockCF.return_value.recommend.return_value  = _make_candidate_set(cf_scores)
            MockCBF.return_value.recommend.return_value = _make_candidate_set(cbf_scores)

            engine = HybridEngine(config=_engine_config(alpha=alpha))
            return engine.recommend(user_id=1, candidate_set=MagicMock(spec=CandidateSet))

    def test_return_type_is_ranked_list(self):
        """recommend() must return a RankedList instance, not a raw list or dict."""
        result = self._call_engine()
        self.assertIsInstance(result, RankedList,
                              f"Expected RankedList, got {type(result).__name__}")

    def test_ranked_list_has_scores_attribute(self):
        """RankedList must expose a .scores mapping."""
        result = self._call_engine()
        self.assertTrue(hasattr(result, "scores"),
                        "RankedList must have a 'scores' attribute")

    def test_ranked_list_has_items_attribute(self):
        """RankedList must expose an ordered .items sequence."""
        result = self._call_engine()
        self.assertTrue(hasattr(result, "items"),
                        "RankedList must have an 'items' attribute")

    def test_items_ordered_by_descending_score(self):
        """Items in RankedList.items must be sorted highest-score-first."""
        result = self._call_engine(alpha=0.5)
        scores = [result.scores[item_id] for item_id in result.items]
        self.assertEqual(
            scores, sorted(scores, reverse=True),
            "RankedList.items must be sorted in descending score order"
        )

    def test_return_is_not_raw_list(self):
        """Return value must not be a plain Python list."""
        result = self._call_engine()
        self.assertNotIsInstance(result, list,
                                 "recommend() must not return a raw list")

    def test_return_is_not_raw_dict(self):
        """Return value must not be a plain Python dict."""
        result = self._call_engine()
        self.assertNotIsInstance(result, dict,
                                 "recommend() must not return a raw dict")