"""
Integration tests for the end-to-end MetadataPipeline.

Run with:
    python manage.py test metadata.tests.test_integration.TestPipelineIntegration

Fixtures live in tests/fixtures/:
    sample.csv   — plain CSV used for most tests
    sample.xlsx  — Excel workbook (sheet: "Sheet1")
    sample.db    — SQLite database with an "orders" table

Only the LLM backend is patched; every other stage runs against real data.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

from metadata.core.pipeline import MetadataPipeline, PipelineResult

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES   = Path(__file__).parent / "fixtures"
CSV_PATH   = FIXTURES / "sample.csv"
EXCEL_PATH = FIXTURES / "sample.xlsx"
DB_PATH    = FIXTURES / "sample.db"

# Column names that must exist in each fixture (adjust to match your files)
CSV_COLUMNS   = ["id", "name", "email", "age", "signup_date", "revenue"]
EXCEL_COLUMNS = CSV_COLUMNS          # same shape as the CSV fixture
SQL_TABLE     = "orders"
SQL_COLUMNS   = ["order_id", "customer_id", "amount", "status"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stub_enrich(profiles: dict) -> dict:
    """Drop-in replacement for LLMGenerator.enrich — no network calls."""
    for col, prof in profiles.items():
        prof.setdefault("description",   f"Auto description for {col}")
        prof.setdefault("tags",          [])
        prof.setdefault("business_name", col)
        prof.setdefault("notes",         "")
    return profiles


def _sql_engine():
    """Return a SQLAlchemy engine pointed at the SQLite fixture."""
    from sqlalchemy import create_engine
    return create_engine(f"sqlite:///{DB_PATH}")


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestPipelineIntegration(TestCase):
    """End-to-end integration tests for MetadataPipeline."""

    # ------------------------------------------------------------------
    # setUp / helpers
    # ------------------------------------------------------------------

    def setUp(self):
        # Verify fixtures exist so failures point at the right problem
        for p in (CSV_PATH, EXCEL_PATH, DB_PATH):
            if not p.exists():
                self.skipTest(f"Fixture not found: {p}")

    def _patch_llm(self):
        return patch(
            "metadata.core.pipeline.LLMGenerator.enrich",
            side_effect=_stub_enrich,
        )

    def _run(self, **kwargs) -> PipelineResult:
        with self._patch_llm():
            return MetadataPipeline(**kwargs).run()

    # ------------------------------------------------------------------
    # 1. CSV — full pipeline smoke test
    # ------------------------------------------------------------------

    def test_csv_returns_pipeline_result(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        self.assertIsInstance(result, PipelineResult)

    def test_csv_profiles_cover_all_columns(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        self.assertEqual(set(result.profiles.keys()), set(CSV_COLUMNS))

    def test_csv_json_schema_is_valid_and_draft07(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        schema = json.loads(result.json_schema)
        self.assertIn("$schema", schema)
        self.assertIn("draft-07", schema["$schema"])

    def test_csv_schema_properties_match_columns(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        props = result.schema.get("properties", {})
        self.assertEqual(set(props.keys()), set(CSV_COLUMNS))

    def test_csv_profiles_have_semantic_type(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        for col, prof in result.profiles.items():
            with self.subTest(column=col):
                self.assertIn("semantic_type", prof)

    def test_csv_profiles_have_llm_enrichment_keys(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        required = {"description", "tags", "business_name", "notes"}
        for col, prof in result.profiles.items():
            with self.subTest(column=col):
                self.assertTrue(required.issubset(prof.keys()))

    def test_csv_stage_times_has_all_six_stages(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        expected = {
            "adapter", "profiler", "extractor",
            "classifier", "llm_generator", "schema_builder",
        }
        self.assertEqual(set(result.stage_times.keys()), expected)

    def test_csv_elapsed_s_is_positive(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        self.assertGreater(result.elapsed_s, 0.0)

    def test_csv_read_kwargs_forwarded(self):
        """Passing read_kwargs (e.g. encoding) must not raise."""
        result = self._run(
            source="csv",
            path=str(CSV_PATH),
            read_kwargs={"encoding": "utf-8"},
        )
        self.assertIsInstance(result, PipelineResult)

    def test_csv_profiles_carry_source_format(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        for col, prof in result.profiles.items():
            with self.subTest(column=col):
                self.assertEqual(prof.get("source_format"), "csv")

    # ------------------------------------------------------------------
    # 2. Excel — full pipeline smoke test
    # ------------------------------------------------------------------

    def test_excel_returns_pipeline_result(self):
        result = self._run(source="excel", path=str(EXCEL_PATH))
        self.assertIsInstance(result, PipelineResult)

    def test_excel_profiles_cover_all_columns(self):
        result = self._run(source="excel", path=str(EXCEL_PATH))
        self.assertEqual(set(result.profiles.keys()), set(EXCEL_COLUMNS))

    def test_excel_json_schema_is_valid(self):
        result = self._run(source="excel", path=str(EXCEL_PATH))
        try:
            json.loads(result.json_schema)
        except json.JSONDecodeError as exc:
            self.fail(f"json_schema is not valid JSON: {exc}")

    def test_excel_schema_and_dict_are_consistent(self):
        result = self._run(source="excel", path=str(EXCEL_PATH))
        self.assertEqual(result.schema, json.loads(result.json_schema))

    def test_excel_profiles_carry_source_format(self):
        result = self._run(source="excel", path=str(EXCEL_PATH))
        for col, prof in result.profiles.items():
            with self.subTest(column=col):
                self.assertEqual(prof.get("source_format"), "excel")

    def test_excel_with_sheet_name_read_kwarg(self):
        """read_kwargs sheet_name should be passed through without error."""
        result = self._run(
            source="excel",
            path=str(EXCEL_PATH),
            read_kwargs={"sheet_name": "Sheet1"},
        )
        self.assertIsInstance(result, PipelineResult)

    # ------------------------------------------------------------------
    # 3. SQL (SQLite fixture) — full pipeline smoke test
    # ------------------------------------------------------------------

    def test_sql_returns_pipeline_result(self):
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
        )
        self.assertIsInstance(result, PipelineResult)

    def test_sql_profiles_cover_all_columns(self):
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
        )
        self.assertEqual(set(result.profiles.keys()), set(SQL_COLUMNS))

    def test_sql_json_schema_properties_match_table(self):
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
        )
        props = result.schema.get("properties", {})
        self.assertEqual(set(props.keys()), set(SQL_COLUMNS))

    def test_sql_custom_query_respected(self):
        """sql_query filters rows; schema columns must still be complete."""
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
            sql_query=f"SELECT * FROM {SQL_TABLE} WHERE status = 'completed'",
        )
        props = result.schema.get("properties", {})
        self.assertEqual(set(props.keys()), set(SQL_COLUMNS))

    def test_sql_dataset_title_defaults_to_table_name(self):
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
        )
        schema = json.loads(result.json_schema)
        self.assertEqual(schema.get("title"), SQL_TABLE)

    def test_sql_schema_report_is_non_empty(self):
        result = self._run(
            source="sql",
            engine=_sql_engine(),
            table_name=SQL_TABLE,
        )
        self.assertTrue(result.schema_report)

    # ------------------------------------------------------------------
    # 4. dataset_title & additional_properties (CSV fixture as base)
    # ------------------------------------------------------------------

    def test_explicit_dataset_title_in_schema(self):
        result = self._run(
            source="csv",
            path=str(CSV_PATH),
            dataset_title="My Dataset",
        )
        self.assertEqual(json.loads(result.json_schema).get("title"), "My Dataset")

    def test_dataset_title_defaults_to_csv_stem(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        self.assertEqual(
            json.loads(result.json_schema).get("title"), CSV_PATH.stem
        )

    def test_additional_properties_false(self):
        result = self._run(
            source="csv",
            path=str(CSV_PATH),
            additional_properties=False,
        )
        self.assertFalse(result.schema.get("additionalProperties", True))

    def test_additional_properties_true_by_default(self):
        result = self._run(source="csv", path=str(CSV_PATH))
        self.assertTrue(result.schema.get("additionalProperties", True))

    # ------------------------------------------------------------------
    # 5. Validation / error paths (no fixtures needed)
    # ------------------------------------------------------------------

    def test_invalid_source_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            MetadataPipeline(source="parquet")
        self.assertIn("parquet", str(ctx.exception))

    def test_csv_missing_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            MetadataPipeline(source="csv").run()

    def test_sql_missing_engine_raises_value_error(self):
        with self.assertRaises(ValueError):
            MetadataPipeline(source="sql", table_name="orders").run()

    def test_sql_missing_table_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            MetadataPipeline(source="sql", engine=MagicMock()).run()

    def test_missing_csv_file_raises_file_not_found(self):
        with self._patch_llm(), self.assertRaises(FileNotFoundError):
            MetadataPipeline(
                source="csv",
                path="/nonexistent/path/data.csv",
            ).run()

    def test_empty_csv_raises_value_error(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("id,name\n")      # header only, no data rows
            tmp = fh.name
        try:
            with self._patch_llm(), self.assertRaises(ValueError) as ctx:
                MetadataPipeline(source="csv", path=tmp).run()
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------------
    # 6. LLM partial-failure resilience (CSV fixture)
    # ------------------------------------------------------------------

    def test_llm_degraded_does_not_raise(self):
        """Generator returning profiles untouched must not break the pipeline."""
        with patch(
            "metadata.core.pipeline.LLMGenerator.enrich",
            side_effect=lambda p: p,
        ):
            result = MetadataPipeline(
                source="csv",
                path=str(CSV_PATH),
            ).run()
        self.assertIsInstance(result, PipelineResult)

    # ------------------------------------------------------------------
    # 7. Cross-source consistency check
    # ------------------------------------------------------------------

    def test_csv_and_excel_schemas_have_same_properties(self):
        """
        Both fixtures share the same columns, so their output schemas must
        have identical property key sets regardless of source format.
        """
        csv_result   = self._run(source="csv",   path=str(CSV_PATH))
        excel_result = self._run(source="excel", path=str(EXCEL_PATH))

        csv_props   = set(csv_result.schema.get("properties", {}).keys())
        excel_props = set(excel_result.schema.get("properties", {}).keys())

        self.assertEqual(csv_props, excel_props)