"""
Tests for DataFrameProfiler (core/profiler.py) and
SemanticClassifier (core/enhancement/semantic_classifier.py).

Run with:
    python manage.py test metadata
"""

from __future__ import annotations

import math
import os
import sys
from unittest import TestCase

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers — adjust to match your Django project layout
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_BASE, "core"))
sys.path.insert(0, os.path.join(_BASE, "core", "enhancement"))

from core.profiler import (
    ColumnProfile,
    DataFrameProfiler,
    SAMPLE_SIZE,
)
from core.enhancement.semantic_classifier import (
    ML_CONFIDENCE_THRESHOLD,
    SEMANTIC_TYPES,
    SemanticClassifier,
    _FeatureExtractor,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _simple_df() -> pd.DataFrame:
    """A small, well-formed DataFrame covering several dtypes."""
    return pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "email":       [f"user{i}@example.com" for i in range(100)],
            "age":         list(range(18, 118)),
            "balance":     [float(i) * 1.5 for i in range(100)],
            "is_active":   [True, False] * 50,
            "signup_date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "notes":       [f"note {i}" for i in range(100)],
        }
    )


def _minimal_profile(col_name: str, dtype: str = "object", **kwargs) -> dict:
    """Bare-minimum profile dict accepted by SemanticClassifier."""
    base = {
        "column_name":   col_name,
        "dtype":         dtype,
        "null_pct":      0.0,
        "null_rate":     0.0,
        "unique_pct":    50.0,
        "unique_rate":   0.5,
        "unique_count":  50,
        "row_count":     100,
        "sample_values": [],
        "is_id_like":    False,
        "min": None, "max": None, "mean": None, "std": None,
    }
    base.update(kwargs)
    return base


# ===========================================================================
# TestColumnProfile
# ===========================================================================

class TestColumnProfile(TestCase):

    def test_default_values(self):
        p = ColumnProfile("test_col")
        self.assertEqual(p.column_name, "test_col")
        self.assertEqual(p.dtype, "")
        self.assertEqual(p.row_count, 0)
        self.assertEqual(p.null_count, 0)
        self.assertEqual(p.null_pct, 0.0)
        self.assertEqual(p.unique_count, 0)
        self.assertEqual(p.unique_pct, 0.0)
        self.assertEqual(p.sample_values, [])
        self.assertFalse(p.is_id_like)
        for attr in ("min", "max", "mean", "median", "std"):
            self.assertIsNone(getattr(p, attr), msg=f"{attr} should default to None")
        for attr in ("min_length", "max_length", "mean_length"):
            self.assertIsNone(getattr(p, attr))
        self.assertIsNone(p.earliest)
        self.assertIsNone(p.latest)

    def test_to_dict_contains_all_slots(self):
        d = ColumnProfile("col").to_dict()
        for slot in ColumnProfile.__slots__:
            self.assertIn(slot, d, msg=f"Slot '{slot}' missing from to_dict()")

    def test_to_dict_none_values_present(self):
        d = ColumnProfile("col").to_dict()
        self.assertIsNone(d["min"])
        self.assertIsNone(d["earliest"])

    def test_repr_contains_column_name(self):
        self.assertIn("my_col", repr(ColumnProfile("my_col")))


# ===========================================================================
# TestDataFrameProfilerConstruction
# ===========================================================================

class TestDataFrameProfilerConstruction(TestCase):

    def test_raises_on_none(self):
        with self.assertRaisesRegex(ValueError, "empty DataFrame"):
            DataFrameProfiler(None)

    def test_raises_on_empty_dataframe(self):
        with self.assertRaisesRegex(ValueError, "empty DataFrame"):
            DataFrameProfiler(pd.DataFrame())

    def test_accepts_single_row(self):
        df = pd.DataFrame({"a": [1]})
        self.assertIs(DataFrameProfiler(df).df, df)


# ===========================================================================
# TestDataFrameProfilerRun
# ===========================================================================

class TestDataFrameProfilerRun(TestCase):

    def setUp(self):
        self.df = _simple_df()
        self.profiles = DataFrameProfiler(self.df).run()

    def test_returns_dict_keyed_by_column(self):
        self.assertEqual(set(self.profiles.keys()), set(self.df.columns))

    def test_each_profile_is_dict(self):
        for col, profile in self.profiles.items():
            self.assertIsInstance(profile, dict, msg=f"Profile for '{col}' is not a dict")

    def test_all_slots_present_in_output(self):
        for col, profile in self.profiles.items():
            for slot in ColumnProfile.__slots__:
                self.assertIn(slot, profile,
                              msg=f"Key '{slot}' missing in profile for '{col}'")

    def test_row_count_correct(self):
        for col, profile in self.profiles.items():
            self.assertEqual(profile["row_count"], len(self.df),
                             msg=f"row_count mismatch for '{col}'")


# ===========================================================================
# TestDataFrameProfilerUniversalFields
# ===========================================================================

class TestDataFrameProfilerUniversalFields(TestCase):

    def setUp(self):
        self.profiles = DataFrameProfiler(_simple_df()).run()

    def test_null_pct_zero_for_clean_column(self):
        self.assertEqual(self.profiles["customer_id"]["null_pct"], 0.0)

    def test_null_pct_correct_for_partial_nulls(self):
        df = pd.DataFrame({"x": [1, None, None, None, None]})
        self.assertEqual(DataFrameProfiler(df).run()["x"]["null_pct"], 80.0)

    def test_null_pct_100_for_all_null_column(self):
        df = pd.DataFrame({"x": [None, None, None]})
        self.assertEqual(DataFrameProfiler(df).run()["x"]["null_pct"], 100.0)

    def test_unique_count(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "b"]})
        self.assertEqual(DataFrameProfiler(df).run()["cat"]["unique_count"], 3)

    def test_unique_pct(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "b"]})
        self.assertEqual(DataFrameProfiler(df).run()["cat"]["unique_pct"], 60.0)

    def test_dtype_int_column(self):
        self.assertIn("int", self.profiles["customer_id"]["dtype"])

    def test_dtype_bool_column(self):
        self.assertIn("bool", self.profiles["is_active"]["dtype"])

    def test_dtype_datetime_column(self):
        self.assertIn("datetime", self.profiles["signup_date"]["dtype"])


# ===========================================================================
# TestDataFrameProfilerSampleValues
# ===========================================================================

class TestDataFrameProfilerSampleValues(TestCase):

    def test_sample_values_max_size(self):
        profiles = DataFrameProfiler(_simple_df()).run()
        for col, profile in profiles.items():
            self.assertLessEqual(len(profile["sample_values"]), SAMPLE_SIZE,
                                 msg=f"sample_values for '{col}' exceeds SAMPLE_SIZE")

    def test_sample_values_empty_for_all_null(self):
        df = pd.DataFrame({"x": pd.Series([None, None], dtype=object)})
        self.assertEqual(DataFrameProfiler(df).run()["x"]["sample_values"], [])

    def test_sample_values_fewer_when_few_distinct(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 2]})
        self.assertEqual(len(DataFrameProfiler(df).run()["x"]["sample_values"]), 2)

    def test_sample_values_are_python_native_types(self):
        df = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int64)})
        p = DataFrameProfiler(df).run()["x"]
        for v in p["sample_values"]:
            self.assertIsInstance(v, (int, float, str, bool, type(None)),
                                  msg=f"Value {v!r} is not a Python-native type")

    def test_numpy_nan_converted_to_none_in_samples(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, 1.0]})
        p = DataFrameProfiler(df).run()["x"]
        for v in p["sample_values"]:
            self.assertTrue(
                v is None or (isinstance(v, float) and not math.isnan(v)),
                msg=f"Unexpected NaN in sample_values: {v!r}",
            )


# ===========================================================================
# TestDataFrameProfilerNumeric
# ===========================================================================

class TestDataFrameProfilerNumeric(TestCase):

    def test_min_max_mean_populated(self):
        df = pd.DataFrame({"n": [1, 2, 3, 4, 5]})
        p = DataFrameProfiler(df).run()["n"]
        self.assertEqual(p["min"], 1.0)
        self.assertEqual(p["max"], 5.0)
        self.assertAlmostEqual(p["mean"], 3.0, places=5)

    def test_median_populated(self):
        df = pd.DataFrame({"n": [1, 2, 3, 4, 5]})
        self.assertEqual(DataFrameProfiler(df).run()["n"]["median"], 3.0)

    def test_std_zero_for_constant_column(self):
        df = pd.DataFrame({"n": [7, 7, 7, 7]})
        self.assertEqual(DataFrameProfiler(df).run()["n"]["std"], 0.0)

    def test_numeric_fields_none_for_string_column(self):
        p = DataFrameProfiler(_simple_df()).run()["email"]
        for attr in ("min", "max", "mean", "median", "std"):
            self.assertIsNone(p[attr])

    def test_nan_excluded_from_stats(self):
        df = pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0]})
        p = DataFrameProfiler(df).run()["n"]
        self.assertEqual(p["min"], 1.0)
        self.assertEqual(p["max"], 4.0)
        self.assertAlmostEqual(p["mean"], 7 / 3, places=4)

    def test_all_null_numeric_column_has_none_stats(self):
        df = pd.DataFrame({"n": pd.array([None, None, None], dtype="Float64")})
        p = DataFrameProfiler(df).run()["n"]
        self.assertIsNone(p["min"])
        self.assertIsNone(p["max"])
        self.assertIsNone(p["mean"])

    def test_inf_values_become_none(self):
        df = pd.DataFrame({"n": [1.0, np.inf, 3.0]})
        self.assertIsNone(DataFrameProfiler(df).run()["n"]["max"])


# ===========================================================================
# TestDataFrameProfilerDatetime
# ===========================================================================

class TestDataFrameProfilerDatetime(TestCase):

    def setUp(self):
        self.p = DataFrameProfiler(_simple_df()).run()["signup_date"]

    def test_earliest_iso_string(self):
        self.assertEqual(self.p["earliest"], "2020-01-01T00:00:00")

    def test_latest_iso_string(self):
        self.assertEqual(self.p["latest"], "2020-04-09T00:00:00")

    def test_datetime_numeric_fields_are_none(self):
        self.assertIsNone(self.p["min"])
        self.assertIsNone(self.p["mean"])

    def test_coerced_datetime_detected(self):
        df = pd.DataFrame({"created_at": ["2021-01-01", "2021-06-15", "2022-03-10"] * 5})
        p = DataFrameProfiler(df).run()["created_at"]
        self.assertIn("datetime", p["dtype"])
        self.assertIsNotNone(p["earliest"])
        self.assertIsNotNone(p["latest"])

    def test_coercion_fails_for_non_date_strings(self):
        df = pd.DataFrame({"x": ["hello", "world", "foo"] * 5})
        p = DataFrameProfiler(df).run()["x"]
        self.assertIsNotNone(p["min_length"])  # fell back to string profiling
        self.assertIsNone(p["earliest"])

    def test_coercion_fails_when_success_rate_below_80pct(self):
        values = ["2021-01-01", "2021-06-15", "2022-03-10"] + ["garbage"] * 7
        df = pd.DataFrame({"x": values})
        self.assertIsNone(DataFrameProfiler(df).run()["x"]["earliest"])


# ===========================================================================
# TestDataFrameProfilerString
# ===========================================================================

class TestDataFrameProfilerString(TestCase):

    def test_min_max_mean_length(self):
        df = pd.DataFrame({"s": ["a", "bb", "ccc"]})
        p = DataFrameProfiler(df).run()["s"]
        self.assertEqual(p["min_length"], 1)
        self.assertEqual(p["max_length"], 3)
        self.assertAlmostEqual(p["mean_length"], 2.0, places=5)

    def test_string_fields_none_for_numeric_column(self):
        p = DataFrameProfiler(_simple_df()).run()["customer_id"]
        self.assertIsNone(p["min_length"])
        self.assertIsNone(p["max_length"])
        self.assertIsNone(p["mean_length"])

    def test_all_null_string_column_has_none_lengths(self):
        df = pd.DataFrame({"s": pd.array([None, None], dtype=object)})
        p = DataFrameProfiler(df).run()["s"]
        self.assertIsNone(p["min_length"])
        self.assertIsNone(p["max_length"])


# ===========================================================================
# TestDataFrameProfilerIsIdLike
# ===========================================================================

class TestDataFrameProfilerIsIdLike(TestCase):

    def _make(self, col_name: str, n: int = 1000):
        return pd.DataFrame({col_name: range(n)})

    def test_id_column_flagged(self):
        self.assertTrue(DataFrameProfiler(self._make("customer_id")).run()["customer_id"]["is_id_like"])

    def test_non_unique_column_not_flagged(self):
        df = pd.DataFrame({"status_id": ["open", "closed"] * 50})
        self.assertFalse(DataFrameProfiler(df).run()["status_id"]["is_id_like"])

    def test_non_id_name_not_flagged(self):
        self.assertFalse(DataFrameProfiler(self._make("price")).run()["price"]["is_id_like"])

    def test_various_id_patterns_flagged(self):
        for col_name in ("id", "uuid", "guid", "order_id", "user_key", "ref_code", "pk"):
            with self.subTest(col_name=col_name):
                p = DataFrameProfiler(self._make(col_name)).run()[col_name]
                self.assertTrue(p["is_id_like"],
                                msg=f"Expected is_id_like=True for '{col_name}'")


# ===========================================================================
# TestDataFrameProfilerResilience
# ===========================================================================

class TestDataFrameProfilerResilience(TestCase):

    def test_single_row_dataframe(self):
        profiles = DataFrameProfiler(pd.DataFrame({"x": [42], "y": ["hello"]})).run()
        self.assertEqual(set(profiles.keys()), {"x", "y"})

    def test_mixed_type_column_does_not_crash(self):
        df = pd.DataFrame({"mixed": [1, "two", 3.0, None, True]})
        self.assertIn("mixed", DataFrameProfiler(df).run())

    def test_high_null_column_profiled(self):
        df = pd.DataFrame({"x": [None] * 99 + [1]})
        self.assertEqual(DataFrameProfiler(df).run()["x"]["null_pct"], 99.0)


# ===========================================================================
# TestSemanticClassifierConstruction
# ===========================================================================

class TestSemanticClassifierConstruction(TestCase):

    def test_default_confidence_threshold(self):
        self.assertEqual(SemanticClassifier().confidence_threshold, ML_CONFIDENCE_THRESHOLD)

    def test_custom_confidence_threshold(self):
        self.assertEqual(SemanticClassifier(confidence_threshold=0.9).confidence_threshold, 0.9)

    def test_model_is_fitted(self):
        self.assertTrue(hasattr(SemanticClassifier()._model, "predict_proba"))

    def test_feature_extractor_present(self):
        self.assertIsInstance(SemanticClassifier()._feature_extractor, _FeatureExtractor)


# ===========================================================================
# TestSemanticClassifierOutputContract
# ===========================================================================

class TestSemanticClassifierOutputContract(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clf = SemanticClassifier()

    def test_classify_injects_semantic_type(self):
        profiles = {"col": _minimal_profile("col")}
        self.clf.classify(profiles)
        self.assertIn("semantic_type", profiles["col"])

    def test_classify_injects_semantic_confidence(self):
        profiles = {"col": _minimal_profile("col")}
        self.clf.classify(profiles)
        self.assertIn("semantic_confidence", profiles["col"])

    def test_confidence_is_float_between_0_and_1(self):
        profiles = {"col": _minimal_profile("col")}
        self.clf.classify(profiles)
        conf = profiles["col"]["semantic_confidence"]
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_semantic_type_is_valid_label(self):
        profiles = {"col": _minimal_profile("col")}
        self.clf.classify(profiles)
        self.assertIn(profiles["col"]["semantic_type"], SEMANTIC_TYPES)

    def test_classify_returns_same_dict_object(self):
        profiles = {"col": _minimal_profile("col")}
        self.assertIs(self.clf.classify(profiles), profiles)

    def test_classify_mutates_inner_dict_in_place(self):
        profile = _minimal_profile("col")
        self.clf.classify({"col": profile})
        self.assertIn("semantic_type", profile)

    def test_column_name_injected_when_missing(self):
        profile = {"dtype": "object", "null_rate": 0.0, "unique_rate": 0.5}
        self.clf.classify({"orphan_col": profile})
        self.assertEqual(profile.get("column_name"), "orphan_col")

    def test_empty_profiles_returns_empty(self):
        self.assertEqual(self.clf.classify({}), {})


# ===========================================================================
# TestSemanticClassifierRules
# ===========================================================================

class TestSemanticClassifierRules(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clf = SemanticClassifier()

    def _classify_one(self, col: str, **kwargs) -> dict:
        profile = {"column_name": col, "dtype": "object",
                   "null_rate": 0.0, "unique_rate": 0.5}
        profile.update(kwargs)
        self.clf.classify({col: profile})
        return profile

    def test_is_primary_key_returns_id_with_full_confidence(self):
        p = self._classify_one("some_col", is_primary_key=True, dtype="int64")
        self.assertEqual(p["semantic_type"], "id")
        self.assertEqual(p["semantic_confidence"], 1.0)

    def test_is_foreign_key_returns_id_with_full_confidence(self):
        p = self._classify_one("order_fk", is_foreign_key=True, dtype="int64")
        self.assertEqual(p["semantic_type"], "id")
        self.assertEqual(p["semantic_confidence"], 1.0)

    def test_is_currency_flag(self):
        self.assertEqual(
            self._classify_one("amount", is_currency=True, dtype="float64")["semantic_type"],
            "currency",
        )

    def test_is_percentage_flag(self):
        self.assertEqual(
            self._classify_one("discount", is_percentage=True, dtype="float64")["semantic_type"],
            "percentage",
        )

    def test_detected_date_format_returns_date(self):
        self.assertEqual(
            self._classify_one("dob", detected_date_format="ISO_DATE")["semantic_type"],
            "date",
        )

    def test_detected_datetime_format_returns_datetime(self):
        self.assertEqual(
            self._classify_one("ts", detected_date_format="ISO_DATETIME")["semantic_type"],
            "datetime",
        )

    def test_bool_dtype_returns_boolean(self):
        self.assertEqual(
            self._classify_one("active", dtype="bool")["semantic_type"],
            "boolean",
        )

    def test_datetime_dtype_returns_datetime(self):
        self.assertEqual(
            self._classify_one("created", dtype="datetime64[ns]")["semantic_type"],
            "datetime",
        )

    def test_primary_key_beats_email_name_rule(self):
        p = self._classify_one("email", is_primary_key=True, dtype="int64")
        self.assertEqual(p["semantic_type"], "id")

    def test_currency_flag_beats_percentage_name_rule(self):
        # 'rate' would resolve to 'percentage' via name rule alone
        p = self._classify_one("rate", is_currency=True, dtype="float64")
        self.assertEqual(p["semantic_type"], "currency")

    def test_name_rules(self):
        cases = [
            ("email",        "email"),
            ("user_email",   "email"),
            ("phone",        "phone"),
            ("mobile",       "phone"),
            ("website",      "url"),
            ("profile_url",  "url"),
            ("latitude",     "latitude"),
            ("lat",          "latitude"),
            ("longitude",    "longitude"),
            ("lng",          "longitude"),
            ("address",      "address"),
            ("street",       "address"),
            ("city",         "address"),
            ("zip",          "address"),
            ("age",          "age"),
            ("year",         "year"),
            ("birth_date",   "date"),
            ("first_name",   "name"),
            ("full_name",    "name"),
            ("description",  "description"),
            ("notes",        "description"),
            ("is_active",    "boolean"),
            ("total_count",  "count"),
            ("score",        "score"),
            ("rating",       "score"),
            ("price",        "currency"),
            ("revenue",      "currency"),
            ("pct",          "percentage"),
            ("category",     "category"),
            ("type",         "category"),
            ("id",           "id"),
            ("user_id",      "id"),
        ]
        for col_name, expected in cases:
            with self.subTest(col=col_name):
                p = self._classify_one(col_name)
                self.assertEqual(p["semantic_type"], expected,
                                 msg=f"Column '{col_name}': expected '{expected}', "
                                     f"got '{p['semantic_type']}'")
                self.assertEqual(p["semantic_confidence"], 1.0)


# ===========================================================================
# TestSemanticClassifierDtypeFallback
# ===========================================================================

class TestSemanticClassifierDtypeFallback(TestCase):

    def setUp(self):
        # threshold=1.0 forces ML to always lose → dtype fallback is always used
        self.clf = SemanticClassifier(confidence_threshold=1.0)

    def _fallback_type(self, dtype_str: str) -> str:
        col = "xyzzy_unknown_col"
        profile = {
            "column_name": col, "dtype": dtype_str,
            "null_rate": 0.0, "unique_rate": 0.5,
            "mean": 500.0, "std": 100.0, "min": 0.0, "max": 999.0,
        }
        self.clf.classify({col: profile})
        return profile["semantic_type"]

    def test_int_dtype_fallback(self):
        self.assertEqual(self._fallback_type("int64"), "count")

    def test_float_dtype_fallback(self):
        self.assertEqual(self._fallback_type("float64"), "score")

    def test_object_dtype_fallback(self):
        self.assertEqual(self._fallback_type("object"), "category")

    def test_bool_dtype_fallback(self):
        self.assertEqual(self._fallback_type("bool"), "boolean")

    def test_datetime_dtype_fallback(self):
        self.assertEqual(self._fallback_type("datetime64[ns]"), "datetime")

    def test_string_dtype_fallback(self):
        self.assertEqual(self._fallback_type("string"), "category")

    def test_unrecognised_dtype_returns_unknown(self):
        self.assertEqual(self._fallback_type("complex128"), "unknown")

    def test_dtype_fallback_mapping_exhaustive(self):
        cases = [
            ("int32",          "count"),
            ("int64",          "count"),
            ("float32",        "score"),
            ("float64",        "score"),
            ("bool",           "boolean"),
            ("datetime64[ns]", "datetime"),
            ("object",         "category"),
            ("string",         "category"),
            ("other_type",     "unknown"),
        ]
        for dtype_str, expected in cases:
            with self.subTest(dtype=dtype_str):
                self.assertEqual(self._fallback_type(dtype_str), expected)


# ===========================================================================
# TestApplyRules
# ===========================================================================

class TestApplyRules(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clf = SemanticClassifier()

    def test_no_rule_fires_for_generic_column(self):
        result = self.clf._apply_rules(
            "xyzzy_42", {"dtype": "object", "null_rate": 0.0, "unique_rate": 0.5}
        )
        self.assertIsNone(result)

    def test_primary_key_beats_name_match(self):
        result = self.clf._apply_rules("email", {"is_primary_key": True, "dtype": "int64"})
        self.assertEqual(result, "id")

    def test_currency_flag_beats_percentage_name(self):
        result = self.clf._apply_rules("rate", {"is_currency": True, "dtype": "float64"})
        self.assertEqual(result, "currency")


# ===========================================================================
# TestTypeDistribution
# ===========================================================================

class TestTypeDistribution(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clf = SemanticClassifier()

    def test_counts_correctly(self):
        dist = self.clf._type_distribution({
            "a": {"semantic_type": "id"},
            "b": {"semantic_type": "email"},
            "c": {"semantic_type": "id"},
        })
        self.assertEqual(dist["id"], 2)
        self.assertEqual(dist["email"], 1)

    def test_missing_semantic_type_counted_as_unknown(self):
        dist = self.clf._type_distribution({"a": {}})
        self.assertEqual(dist.get("unknown", 0), 1)

    def test_sorted_by_count_descending(self):
        dist = self.clf._type_distribution({
            "a": {"semantic_type": "id"},
            "b": {"semantic_type": "id"},
            "c": {"semantic_type": "email"},
        })
        counts = list(dist.values())
        self.assertEqual(counts, sorted(counts, reverse=True))


# ===========================================================================
# TestFeatureExtractor
# ===========================================================================

class TestFeatureExtractor(TestCase):

    def setUp(self):
        self.fe = _FeatureExtractor()

    def _vec(self, col_name="x", dtype="object", **kwargs):
        return self.fe.transform({"column_name": col_name, "dtype": dtype, **kwargs})

    def test_output_shape(self):
        self.assertEqual(self._vec().shape, (_FeatureExtractor.N_FEATURES,))

    def test_output_dtype_float32(self):
        self.assertEqual(self._vec().dtype, np.float32)

    def test_int_dtype_flag(self):
        v = self._vec(dtype="int64")
        self.assertEqual(v[3], 1.0)
        self.assertEqual(v[4], 0.0)

    def test_float_dtype_flag(self):
        v = self._vec(dtype="float64")
        self.assertEqual(v[4], 1.0)
        self.assertEqual(v[3], 0.0)

    def test_bool_dtype_flag(self):
        self.assertEqual(self._vec(dtype="bool")[5], 1.0)

    def test_datetime_dtype_flag(self):
        self.assertEqual(self._vec(dtype="datetime64[ns]")[6], 1.0)

    def test_object_dtype_flag(self):
        self.assertEqual(self._vec(dtype="object")[7], 1.0)

    def test_name_with_digit_flag(self):
        self.assertEqual(self._vec(col_name="col1")[0], 1.0)

    def test_name_without_digit_flag(self):
        self.assertEqual(self._vec(col_name="col")[0], 0.0)

    def test_is_currency_flag(self):
        self.assertEqual(self._vec(dtype="float64", is_currency=True)[15], 1.0)

    def test_is_percentage_flag(self):
        self.assertEqual(self._vec(dtype="float64", is_percentage=True)[16], 1.0)

    def test_detected_date_format_flag(self):
        self.assertEqual(self._vec(detected_date_format="ISO_DATE")[17], 1.0)

    def test_null_rate_clamped_to_1(self):
        self.assertEqual(self._vec(null_rate=2.0)[8], 1.0)

    def test_max_gt_million_flag(self):
        self.assertEqual(self._vec(dtype="float64", max=5_000_000.0)[22], 1.0)

    def test_min_negative_flag(self):
        self.assertEqual(self._vec(dtype="float64", min=-1.0)[20], 1.0)

    def test_empty_profile_does_not_crash(self):
        v = self.fe.transform({})
        self.assertEqual(v.shape, (_FeatureExtractor.N_FEATURES,))


# ===========================================================================
# TestEndToEndIntegration
# ===========================================================================

class TestEndToEndIntegration(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.clf = SemanticClassifier()
        profiles = DataFrameProfiler(_simple_df()).run()
        for prof in profiles.values():
            prof.setdefault("null_rate",   prof["null_pct"]   / 100)
            prof.setdefault("unique_rate", prof["unique_pct"] / 100)
        cls.profiles = cls.clf.classify(profiles)

    def test_email_column_classified(self):
        self.assertEqual(self.profiles["email"]["semantic_type"], "email")

    def test_id_column_classified(self):
        self.assertEqual(self.profiles["customer_id"]["semantic_type"], "id")

    def test_all_columns_have_valid_semantic_type(self):
        for col, prof in self.profiles.items():
            with self.subTest(col=col):
                self.assertIn(prof["semantic_type"], SEMANTIC_TYPES,
                              msg=f"Column '{col}' got invalid type '{prof['semantic_type']}'")

    def test_all_columns_have_valid_confidence(self):
        for col, prof in self.profiles.items():
            with self.subTest(col=col):
                conf = prof["semantic_confidence"]
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)