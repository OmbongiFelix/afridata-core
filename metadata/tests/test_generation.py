"""Run with:
    python manage.py test metadata

Tests for:
    - core/enhancement/llm_generator.py  → TestLLMGenerator
    - core/schema_builder.py             → TestSchemaBuilder
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Minimal Django settings bootstrap — keeps tests runnable without a full
# Django project on the path.
# ---------------------------------------------------------------------------
def _bootstrap_django(extra_settings: dict | None = None):
    """Configure a minimal in-memory Django settings object."""
    try:
        from django.conf import settings as dj_settings
        if not dj_settings.configured:
            raise RuntimeError("not configured")
    except Exception:
        import django
        from django.conf import settings as dj_settings
        base = dict(
            USE_TZ=True,
            DATABASES={},
            INSTALLED_APPS=[],
            LLM_BACKEND="openai",
            OPENAI_API_KEY="test-key",
        )
        if extra_settings:
            base.update(extra_settings)
        dj_settings.configure(**base)


_bootstrap_django()


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def _make_mock_backend(return_value: str = "{}") -> MagicMock:
    """Return a mock _LLMBackend whose complete() returns *return_value*."""
    backend = MagicMock()
    backend.complete.return_value = return_value
    return backend


def _minimal_profile(
    *,
    semantic_type: str = "name",
    dtype: str = "object",
    is_nullable: bool = True,
    null_pct: float = 0.0,
    description: str = "A column description.",
    business_name: str = "My Column",
    tags: list | None = None,
    notes: str = "",
    **extra,
) -> dict:
    return {
        "semantic_type": semantic_type,
        "dtype": dtype,
        "is_nullable": is_nullable,
        "null_pct": null_pct,
        "description": description,
        "business_name": business_name,
        "tags": tags or [],
        "notes": notes,
        **extra,
    }


# ===========================================================================
# TestLLMGenerator
# ===========================================================================

class TestLLMGenerator(unittest.TestCase):
    """Unit tests for core/enhancement/llm_generator.py"""

    # ------------------------------------------------------------------ #
    # Imports & module-level helpers                                       #
    # ------------------------------------------------------------------ #

    def setUp(self):
        from metadata.core.enhancement import llm_generator as mod
        self.mod = mod

    # ------------------------------------------------------------------ #
    # _chunk                                                               #
    # ------------------------------------------------------------------ #

    def test_chunk_even_split(self):
        result = self.mod._chunk([1, 2, 3, 4], 2)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_chunk_uneven_split(self):
        result = self.mod._chunk([1, 2, 3, 4, 5], 2)
        self.assertEqual(result, [[1, 2], [3, 4], [5]])

    def test_chunk_size_larger_than_list(self):
        result = self.mod._chunk([1, 2], 10)
        self.assertEqual(result, [[1, 2]])

    def test_chunk_empty_list(self):
        self.assertEqual(self.mod._chunk([], 5), [])

    # ------------------------------------------------------------------ #
    # _default_business_name                                               #
    # ------------------------------------------------------------------ #

    def test_default_business_name_snake_case(self):
        self.assertEqual(self.mod._default_business_name("customer_id"), "Customer Id")

    def test_default_business_name_hyphen(self):
        self.assertEqual(self.mod._default_business_name("first-name"), "First Name")

    def test_default_business_name_plain(self):
        self.assertEqual(self.mod._default_business_name("email"), "Email")

    # ------------------------------------------------------------------ #
    # _safe_defaults                                                       #
    # ------------------------------------------------------------------ #

    def test_safe_defaults_keys(self):
        result = self.mod._safe_defaults("order_total")
        self.assertIn("description", result)
        self.assertIn("tags", result)
        self.assertIn("business_name", result)
        self.assertIn("notes", result)

    def test_safe_defaults_tags_empty(self):
        self.assertEqual(self.mod._safe_defaults("col")["tags"], [])

    def test_safe_defaults_notes_contains_unavailable(self):
        self.assertIn("unavailable", self.mod._safe_defaults("col")["notes"].lower())

    def test_safe_defaults_business_name_derived_from_col(self):
        self.assertEqual(self.mod._safe_defaults("user_name")["business_name"], "User Name")

    # ------------------------------------------------------------------ #
    # _sanitise_tags                                                       #
    # ------------------------------------------------------------------ #

    def test_sanitise_tags_filters_unknown(self):
        result = self.mod._sanitise_tags(["PII", "not_a_real_tag", "financial"])
        self.assertEqual(result, ["PII", "financial"])

    def test_sanitise_tags_non_list_returns_empty(self):
        self.assertEqual(self.mod._sanitise_tags("PII"), [])
        self.assertEqual(self.mod._sanitise_tags(None), [])

    def test_sanitise_tags_all_allowed(self):
        tags = ["PII", "temporal", "metric"]
        self.assertEqual(self.mod._sanitise_tags(tags), tags)

    def test_sanitise_tags_empty_list(self):
        self.assertEqual(self.mod._sanitise_tags([]), [])

    # ------------------------------------------------------------------ #
    # _build_system_prompt                                                 #
    # ------------------------------------------------------------------ #

    def test_build_system_prompt_contains_allowed_tags(self):
        prompt = self.mod._build_system_prompt()
        self.assertIn("PII", prompt)
        self.assertIn("financial", prompt)

    def test_build_system_prompt_is_string(self):
        self.assertIsInstance(self.mod._build_system_prompt(), str)

    # ------------------------------------------------------------------ #
    # _build_user_prompt                                                   #
    # ------------------------------------------------------------------ #

    def test_build_user_prompt_includes_column_name(self):
        batch = {"email": {"dtype": "object", "semantic_type": "email"}}
        prompt = self.mod._build_user_prompt(batch)
        self.assertIn("email", prompt)

    def test_build_user_prompt_count_placeholder(self):
        batch = {"a": {}, "b": {}}
        prompt = self.mod._build_user_prompt(batch)
        self.assertIn("2", prompt)

    # ------------------------------------------------------------------ #
    # _parse_llm_response                                                  #
    # ------------------------------------------------------------------ #

    def test_parse_valid_response(self):
        raw = json.dumps({
            "col_a": {
                "description": "Some desc",
                "tags": ["PII"],
                "business_name": "Col A",
                "notes": "",
            }
        })
        result = self.mod._parse_llm_response(raw, ["col_a"])
        self.assertEqual(result["col_a"]["description"], "Some desc")
        self.assertEqual(result["col_a"]["tags"], ["PII"])

    def test_parse_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "col_a": {"description": "D", "tags": [], "business_name": "B", "notes": ""}
        }) + "\n```"
        result = self.mod._parse_llm_response(raw, ["col_a"])
        self.assertEqual(result["col_a"]["description"], "D")

    def test_parse_missing_column_uses_defaults(self):
        raw = json.dumps({"other_col": {"description": "D", "tags": [], "business_name": "B", "notes": ""}})
        result = self.mod._parse_llm_response(raw, ["missing_col"])
        self.assertIn("missing_col", result)
        self.assertEqual(result["missing_col"]["tags"], [])

    def test_parse_invalid_json_raises_parse_error(self):
        with self.assertRaises(self.mod.LLMParseError):
            self.mod._parse_llm_response("not-json", ["col"])

    def test_parse_non_object_raises_parse_error(self):
        with self.assertRaises(self.mod.LLMParseError):
            self.mod._parse_llm_response(json.dumps([1, 2, 3]), ["col"])

    def test_parse_filters_disallowed_tags(self):
        raw = json.dumps({
            "col": {"description": "", "tags": ["PII", "INVALID_TAG"], "business_name": "", "notes": ""}
        })
        result = self.mod._parse_llm_response(raw, ["col"])
        self.assertNotIn("INVALID_TAG", result["col"]["tags"])
        self.assertIn("PII", result["col"]["tags"])

    def test_parse_extra_columns_ignored(self):
        raw = json.dumps({
            "expected": {"description": "E", "tags": [], "business_name": "E", "notes": ""},
            "extra":    {"description": "X", "tags": [], "business_name": "X", "notes": ""},
        })
        result = self.mod._parse_llm_response(raw, ["expected"])
        self.assertNotIn("extra", result)

    # ------------------------------------------------------------------ #
    # _default_model                                                       #
    # ------------------------------------------------------------------ #

    def test_default_model_openai(self):
        self.assertEqual(self.mod._default_model("openai"), "gpt-4o-mini")

    def test_default_model_anthropic(self):
        self.assertIn("claude", self.mod._default_model("anthropic").lower())

    def test_default_model_unknown_falls_back(self):
        result = self.mod._default_model("unknown_backend")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    # ------------------------------------------------------------------ #
    # _with_retries                                                        #
    # ------------------------------------------------------------------ #

    def test_with_retries_success_first_attempt(self):
        fn = MagicMock(return_value="ok")
        result = self.mod._with_retries(fn, max_retries=3, backoff_base=0,
                                         logger_=MagicMock(), label="test")
        self.assertEqual(result, "ok")
        fn.assert_called_once()

    def test_with_retries_succeeds_on_second_attempt(self):
        fn = MagicMock(side_effect=[self.mod.LLMRequestError("fail"), "ok"])
        with patch("time.sleep"):
            result = self.mod._with_retries(fn, max_retries=3, backoff_base=2,
                                             logger_=MagicMock(), label="test")
        self.assertEqual(result, "ok")
        self.assertEqual(fn.call_count, 2)

    def test_with_retries_all_fail_raises(self):
        fn = MagicMock(side_effect=self.mod.LLMRequestError("always fails"))
        with patch("time.sleep"):
            with self.assertRaises(self.mod.LLMRequestError):
                self.mod._with_retries(fn, max_retries=3, backoff_base=2,
                                        logger_=MagicMock(), label="test")
        self.assertEqual(fn.call_count, 3)

    # ------------------------------------------------------------------ #
    # LLMGenerator.enrich — happy path                                    #
    # ------------------------------------------------------------------ #

    def _make_generator(self, backend_return: str = "{}") -> "LLMGenerator":
        """Build an LLMGenerator with a mock backend."""
        from metadata.core.enhancement.llm_generator import LLMGenerator
        gen = object.__new__(LLMGenerator)
        gen.batch_size = 10
        gen.max_retries = 1
        gen.retry_backoff = 0.0
        gen._backend = _make_mock_backend(backend_return)
        gen._system_prompt = "sys"
        gen.logger = MagicMock()
        return gen

    def test_enrich_empty_profiles_returns_unchanged(self):
        gen = self._make_generator()
        result = gen.enrich({})
        self.assertEqual(result, {})

    def test_enrich_injects_enrichment_keys(self):
        payload = {
            "email": {"description": "D", "tags": ["PII"],
                      "business_name": "Email", "notes": ""}
        }
        gen = self._make_generator(backend_return=json.dumps(payload))
        profiles = {"email": {"dtype": "object"}}
        result = gen.enrich(profiles)
        self.assertIn("description", result["email"])
        self.assertIn("tags", result["email"])
        self.assertIn("business_name", result["email"])

    def test_enrich_applies_safe_defaults_on_request_error(self):
        from metadata.core.enhancement.llm_generator import LLMGenerator, LLMRequestError
        gen = self._make_generator()
        gen._backend.complete.side_effect = LLMRequestError("boom")
        profiles = {"col": {"dtype": "object"}}
        result = gen.enrich(profiles)
        # Should not raise; safe defaults applied
        self.assertIn("notes", result["col"])
        self.assertIn("unavailable", result["col"]["notes"].lower())

    def test_enrich_applies_safe_defaults_on_parse_error(self):
        gen = self._make_generator(backend_return="not valid json")
        profiles = {"col": {"dtype": "object"}}
        result = gen.enrich(profiles)
        self.assertIn("notes", result["col"])

    def test_enrich_batches_columns(self):
        """Generator must split columns into batches of batch_size."""
        payload = {f"col_{i}": {"description": "d", "tags": [],
                                 "business_name": f"Col {i}", "notes": ""}
                   for i in range(5)}
        gen = self._make_generator(backend_return=json.dumps(payload))
        gen.batch_size = 2
        profiles = {f"col_{i}": {"dtype": "object"} for i in range(5)}
        gen.enrich(profiles)
        # 5 columns / batch_size 2 → 3 batches → 3 backend calls
        self.assertEqual(gen._backend.complete.call_count, 3)

    def test_enrich_returns_same_profiles_dict(self):
        payload = {"col": {"description": "D", "tags": [], "business_name": "C", "notes": ""}}
        gen = self._make_generator(backend_return=json.dumps(payload))
        profiles = {"col": {"dtype": "object"}}
        result = gen.enrich(profiles)
        self.assertIs(result, profiles)

    # ------------------------------------------------------------------ #
    # LLMGenerator.enrichment_report                                      #
    # ------------------------------------------------------------------ #

    def test_enrichment_report_counts_enriched(self):
        from metadata.core.enhancement.llm_generator import LLMGenerator
        gen = object.__new__(LLMGenerator)
        gen.logger = MagicMock()
        profiles = {
            "a": {"description": "Has description", "tags": ["PII"]},
            "b": {"description": "", "tags": []},
        }
        report = gen.enrichment_report(profiles)
        self.assertEqual(report["total_columns"], 2)
        self.assertEqual(report["enriched_columns"], 1)
        self.assertIn("b", report["unenriched_columns"])

    def test_enrichment_report_tag_frequency(self):
        from metadata.core.enhancement.llm_generator import LLMGenerator
        gen = object.__new__(LLMGenerator)
        gen.logger = MagicMock()
        profiles = {
            "a": {"description": "D", "tags": ["PII", "financial"]},
            "b": {"description": "D", "tags": ["PII"]},
        }
        report = gen.enrichment_report(profiles)
        self.assertEqual(report["tag_frequency"]["PII"], 2)
        self.assertEqual(report["tag_frequency"]["financial"], 1)

    def test_enrichment_report_empty_profiles(self):
        from metadata.core.enhancement.llm_generator import LLMGenerator
        gen = object.__new__(LLMGenerator)
        gen.logger = MagicMock()
        report = gen.enrichment_report({})
        self.assertEqual(report["total_columns"], 0)
        self.assertEqual(report["enriched_columns"], 0)

    # ------------------------------------------------------------------ #
    # Custom exceptions hierarchy                                         #
    # ------------------------------------------------------------------ #

    def test_exception_hierarchy(self):
        mod = self.mod
        self.assertTrue(issubclass(mod.LLMConfigError, mod.LLMError))
        self.assertTrue(issubclass(mod.LLMParseError, mod.LLMError))
        self.assertTrue(issubclass(mod.LLMRequestError, mod.LLMError))


# ===========================================================================
# TestSchemaBuilder
# ===========================================================================

class TestSchemaBuilder(unittest.TestCase):
    """Unit tests for core/schema_builder.py"""

    def setUp(self):
        from metadata.core import schema_builder as mod
        self.mod = mod
        self.SchemaBuilder = mod.SchemaBuilder

    # ------------------------------------------------------------------ #
    # _dtype_to_json_type                                                  #
    # ------------------------------------------------------------------ #

    def test_dtype_int64(self):
        self.assertEqual(self.mod._dtype_to_json_type("int64"), "integer")

    def test_dtype_float32(self):
        self.assertEqual(self.mod._dtype_to_json_type("float32"), "number")

    def test_dtype_bool(self):
        self.assertEqual(self.mod._dtype_to_json_type("bool"), "boolean")

    def test_dtype_object(self):
        self.assertEqual(self.mod._dtype_to_json_type("object"), "string")

    def test_dtype_datetime(self):
        self.assertEqual(self.mod._dtype_to_json_type("datetime64[ns]"), "string")

    def test_dtype_unknown_falls_back_to_string(self):
        self.assertEqual(self.mod._dtype_to_json_type("completely_unknown"), "string")

    # ------------------------------------------------------------------ #
    # _resolve_json_type                                                   #
    # ------------------------------------------------------------------ #

    def test_resolve_json_type_uses_semantic_type(self):
        profile = {"semantic_type": "email", "dtype": "int64"}
        # semantic_type "email" → "string" overrides dtype "int64" → "integer"
        self.assertEqual(self.mod._resolve_json_type(profile), "string")

    def test_resolve_json_type_falls_back_to_dtype(self):
        profile = {"semantic_type": "not_in_map", "dtype": "float64"}
        self.assertEqual(self.mod._resolve_json_type(profile), "number")

    def test_resolve_json_type_unknown_semantic_falls_back(self):
        profile = {"semantic_type": "unknown", "dtype": "object"}
        # "unknown" IS in _SEMANTIC_TYPE_MAP → "string"
        self.assertEqual(self.mod._resolve_json_type(profile), "string")

    def test_resolve_json_type_no_keys_returns_string(self):
        self.assertEqual(self.mod._resolve_json_type({}), "string")

    # ------------------------------------------------------------------ #
    # _resolve_format                                                      #
    # ------------------------------------------------------------------ #

    def test_resolve_format_email(self):
        self.assertEqual(self.mod._resolve_format({"semantic_type": "email"}), "email")

    def test_resolve_format_url(self):
        self.assertEqual(self.mod._resolve_format({"semantic_type": "url"}), "uri")

    def test_resolve_format_date(self):
        self.assertEqual(self.mod._resolve_format({"semantic_type": "date"}), "date")

    def test_resolve_format_id_string_type(self):
        profile = {"semantic_type": "id", "dtype": "object"}
        self.assertEqual(self.mod._resolve_format(profile), "uuid")

    def test_resolve_format_id_numeric_suppressed(self):
        # id with integer dtype → no uuid format
        profile = {"semantic_type": "id", "dtype": "int64"}
        self.assertIsNone(self.mod._resolve_format(profile))

    def test_resolve_format_no_format_for_name(self):
        self.assertIsNone(self.mod._resolve_format({"semantic_type": "name"}))

    # ------------------------------------------------------------------ #
    # _build_column_schema                                                 #
    # ------------------------------------------------------------------ #

    def test_build_column_schema_basic_keys(self):
        profile = _minimal_profile(semantic_type="name", dtype="object",
                                   is_nullable=False, null_pct=0.0)
        schema = self.mod._build_column_schema(profile)
        self.assertIn("type", schema)
        self.assertIn("description", schema)
        self.assertIn("title", schema)
        self.assertIn("x-semantic-type", schema)
        self.assertIn("x-tags", schema)

    def test_build_column_schema_nullable_type_is_list(self):
        profile = _minimal_profile(is_nullable=True)
        schema = self.mod._build_column_schema(profile)
        self.assertIsInstance(schema["type"], list)
        self.assertIn("null", schema["type"])

    def test_build_column_schema_non_nullable_type_is_string(self):
        profile = _minimal_profile(is_nullable=False, null_pct=0.0)
        schema = self.mod._build_column_schema(profile)
        self.assertIsInstance(schema["type"], str)

    def test_build_column_schema_numeric_constraints(self):
        profile = _minimal_profile(semantic_type="score", dtype="float64",
                                   is_nullable=False, null_pct=0.0,
                                   min=0.0, max=100.0)
        schema = self.mod._build_column_schema(profile)
        self.assertEqual(schema["minimum"], 0.0)
        self.assertEqual(schema["maximum"], 100.0)

    def test_build_column_schema_string_length_constraints(self):
        profile = _minimal_profile(semantic_type="name", dtype="object",
                                   is_nullable=False, null_pct=0.0,
                                   min_length=1, max_length=255)
        schema = self.mod._build_column_schema(profile)
        self.assertEqual(schema["minLength"], 1)
        self.assertEqual(schema["maxLength"], 255)

    def test_build_column_schema_enum_low_cardinality(self):
        profile = _minimal_profile(
            semantic_type="category", dtype="object",
            is_nullable=False, null_pct=0.0,
            unique_count=3,
            sample_values=["A", "B", "C"],
        )
        schema = self.mod._build_column_schema(profile)
        self.assertIn("enum", schema)
        self.assertEqual(set(schema["enum"]), {"A", "B", "C"})

    def test_build_column_schema_examples_when_no_enum(self):
        profile = _minimal_profile(
            semantic_type="name", dtype="object",
            is_nullable=False, null_pct=0.0,
            sample_values=["Alice", "Bob", "Charlie"],
        )
        schema = self.mod._build_column_schema(profile)
        self.assertIn("examples", schema)
        self.assertNotIn("enum", schema)

    def test_build_column_schema_primary_key_extension(self):
        profile = _minimal_profile(is_nullable=False, null_pct=0.0,
                                   is_primary_key=True)
        schema = self.mod._build_column_schema(profile)
        self.assertTrue(schema.get("x-primary-key"))

    def test_build_column_schema_foreign_key_extension(self):
        profile = _minimal_profile(is_nullable=False, null_pct=0.0,
                                   is_foreign_key=True,
                                   fk_references=["orders.id"])
        schema = self.mod._build_column_schema(profile)
        self.assertTrue(schema.get("x-foreign-key"))
        self.assertIn("orders.id", schema.get("x-fk-references", []))

    def test_build_column_schema_notes_omitted_when_empty(self):
        profile = _minimal_profile(notes="")
        schema = self.mod._build_column_schema(profile)
        self.assertNotIn("x-notes", schema)

    def test_build_column_schema_notes_present_when_set(self):
        profile = _minimal_profile(notes="Check for nulls")
        schema = self.mod._build_column_schema(profile)
        self.assertIn("x-notes", schema)

    # ------------------------------------------------------------------ #
    # SchemaBuilder.__init__                                               #
    # ------------------------------------------------------------------ #

    def test_init_defaults(self):
        builder = self.SchemaBuilder()
        self.assertEqual(builder.dataset_title, "Dataset")
        self.assertEqual(builder.dataset_description, "")
        self.assertTrue(builder.additional_properties)

    def test_init_custom_values(self):
        builder = self.SchemaBuilder(
            dataset_title="Orders",
            dataset_description="Order data",
            additional_properties=False,
        )
        self.assertEqual(builder.dataset_title, "Orders")
        self.assertFalse(builder.additional_properties)

    # ------------------------------------------------------------------ #
    # SchemaBuilder.build                                                  #
    # ------------------------------------------------------------------ #

    def test_build_raises_on_empty_profiles(self):
        builder = self.SchemaBuilder(dataset_title="T")
        with self.assertRaises(ValueError):
            builder.build({})

    def test_build_top_level_keys(self):
        builder = self.SchemaBuilder(dataset_title="MyData")
        profiles = {"col": _minimal_profile()}
        schema = builder.build(profiles)
        self.assertEqual(schema["$schema"], self.mod.JSON_SCHEMA_DRAFT)
        self.assertEqual(schema["title"], "MyData")
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)

    def test_build_description_present_when_provided(self):
        builder = self.SchemaBuilder(dataset_title="T", dataset_description="Desc")
        schema = builder.build({"col": _minimal_profile()})
        self.assertEqual(schema["description"], "Desc")

    def test_build_description_absent_when_empty(self):
        builder = self.SchemaBuilder(dataset_title="T")
        schema = builder.build({"col": _minimal_profile()})
        self.assertNotIn("description", schema)

    def test_build_required_columns(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "mandatory": _minimal_profile(is_nullable=False, null_pct=0.0),
            "optional":  _minimal_profile(is_nullable=True,  null_pct=0.1),
        }
        schema = builder.build(profiles)
        self.assertIn("required", schema)
        self.assertIn("mandatory", schema["required"])
        self.assertNotIn("optional", schema.get("required", []))

    def test_build_no_required_key_when_all_nullable(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {"col": _minimal_profile(is_nullable=True)}
        schema = builder.build(profiles)
        self.assertNotIn("required", schema)

    def test_build_additional_properties_false(self):
        builder = self.SchemaBuilder(dataset_title="T", additional_properties=False)
        schema = builder.build({"col": _minimal_profile()})
        self.assertFalse(schema["additionalProperties"])

    def test_build_multiple_columns(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "email":  _minimal_profile(semantic_type="email"),
            "amount": _minimal_profile(semantic_type="currency", dtype="float64"),
        }
        schema = builder.build(profiles)
        self.assertIn("email",  schema["properties"])
        self.assertIn("amount", schema["properties"])

    # ------------------------------------------------------------------ #
    # SchemaBuilder.to_json / from_json                                   #
    # ------------------------------------------------------------------ #

    def test_to_json_returns_valid_json_string(self):
        builder = self.SchemaBuilder(dataset_title="T")
        schema = builder.build({"col": _minimal_profile()})
        json_str = self.SchemaBuilder.to_json(schema)
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)

    def test_to_json_compact_when_indent_none(self):
        builder = self.SchemaBuilder(dataset_title="T")
        schema = builder.build({"col": _minimal_profile()})
        compact = self.SchemaBuilder.to_json(schema, indent=None)
        self.assertNotIn("\n", compact)

    def test_from_json_round_trips(self):
        builder = self.SchemaBuilder(dataset_title="Orders")
        schema = builder.build({"col": _minimal_profile()})
        json_str = self.SchemaBuilder.to_json(schema)
        restored = self.SchemaBuilder.from_json(json_str)
        self.assertEqual(schema["title"], restored["title"])
        self.assertIn("col", restored["properties"])

    def test_from_json_invalid_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            self.SchemaBuilder.from_json("not json at all")

    # ------------------------------------------------------------------ #
    # SchemaBuilder.schema_report                                         #
    # ------------------------------------------------------------------ #

    def test_schema_report_basic_counts(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "a": _minimal_profile(is_nullable=False, null_pct=0.0, tags=["PII"]),
            "b": _minimal_profile(is_nullable=True),
        }
        schema = builder.build(profiles)
        report = builder.schema_report(schema)
        self.assertEqual(report["total_properties"], 2)
        self.assertEqual(report["required_count"], 1)

    def test_schema_report_nullable_count(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "a": _minimal_profile(is_nullable=True),
            "b": _minimal_profile(is_nullable=False, null_pct=0.0),
        }
        schema = builder.build(profiles)
        report = builder.schema_report(schema)
        self.assertEqual(report["nullable_count"], 1)

    def test_schema_report_tagged_columns(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "a": _minimal_profile(tags=["PII", "sensitive"]),
            "b": _minimal_profile(tags=[]),
        }
        schema = builder.build(profiles)
        report = builder.schema_report(schema)
        self.assertIn("a", report["tagged_columns"])
        self.assertNotIn("b", report["tagged_columns"])

    def test_schema_report_type_distribution(self):
        builder = self.SchemaBuilder(dataset_title="T")
        profiles = {
            "e": _minimal_profile(semantic_type="email",    dtype="object",  is_nullable=False, null_pct=0.0),
            "n": _minimal_profile(semantic_type="currency", dtype="float64", is_nullable=False, null_pct=0.0),
        }
        schema = builder.build(profiles)
        report = builder.schema_report(schema)
        type_dist = report["type_distribution"]
        # Both non-nullable → raw type strings
        self.assertIn("string",  type_dist)
        self.assertIn("number",  type_dist)

    def test_schema_report_empty_schema(self):
        builder = self.SchemaBuilder(dataset_title="T")
        schema = {"$schema": self.mod.JSON_SCHEMA_DRAFT, "title": "T",
                  "type": "object", "properties": {}, "additionalProperties": True}
        report = builder.schema_report(schema)
        self.assertEqual(report["total_properties"], 0)
        self.assertEqual(report["required_count"], 0)


# ===========================================================================
# Entry-point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()