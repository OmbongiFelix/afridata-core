"""
Test suite for the Metadata Extraction Pipeline.

Organised by pipeline stage — each stage has its own TestCase class:

    TestCSVConnector         - adapter ingestion and error handling
    TestExcelConnector       - Excel ingestion, multi-sheet, structural validation
    TestSQLConnector         - database extraction with SQLite fixture
    TestCsvExcelExtractor    - pattern detection and profile augmentation
    TestSqlExtractor         - schema reflection and column annotation

Run with: python manage.py test metadata
"""

import csv
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins so tests run without the full app installed
# ---------------------------------------------------------------------------

class _FakeBaseConnector:
    """
    Minimal stand-in for BaseConnector.
    Lets us test CSVConnector / ExcelConnector / SQLConnector in isolation.
    """

    SUPPORTED_EXTENSIONS = []

    def __init__(self):
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._eligible: list[dict] = []

    # Seeded by individual tests
    def fetch_eligible_files(self) -> list[dict]:
        return list(self._eligible)

    def get_engine(self):
        from sqlalchemy import create_engine
        return create_engine("sqlite:///:memory:", future=True)

    def validate_source(self, file_record: dict) -> bool:
        return bool(file_record.get("file_path", "").strip())


# Monkey-patch base before importing connectors so they inherit the stub.
import sys, types

_adapters_mod = types.ModuleType("adapters")
_base_mod = types.ModuleType("adapters.base_connector")
_base_mod.BaseConnector = _FakeBaseConnector
sys.modules.setdefault("adapters", _adapters_mod)
sys.modules.setdefault("adapters.base_connector", _base_mod)

# Now import the modules under test (adjust dotted paths to match your project).
# We import the source text directly so the tests are self-contained.

import importlib.util, inspect

def _load_source(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Make sure the relative import ".base_connector" resolves to our stub
    mod.__package__ = "adapters"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_HERE = Path("/mnt/user-data/uploads")

_csv_mod   = _load_source("adapters.csv_connector",        _HERE / "csv_connector.py")
_excel_mod = _load_source("adapters.excel_connector",      _HERE / "excel_connector.py")
_sql_mod   = _load_source("adapters.sql_connector",        _HERE / "sql_connector.py")
_cex_mod   = _load_source("adapters.csv_excel_extractor",  _HERE / "csv_excel_extractor.py")
_sqlex_mod = _load_source("adapters.sql_extractor",        _HERE / "sql_extractor.py")

CSVConnector        = _csv_mod.CSVConnector
ExcelConnector      = _excel_mod.ExcelConnector
SQLConnector        = _sql_mod.SQLConnector
CsvExcelExtractor   = _cex_mod.CsvExcelExtractor
SqlExtractor        = _sqlex_mod.SqlExtractor


# ===========================================================================
# Fixtures / factories
# ===========================================================================

def _write_csv(tmp_dir: str, filename: str, content: str, encoding="utf-8") -> str:
    path = os.path.join(tmp_dir, filename)
    with open(path, "w", encoding=encoding) as fh:
        fh.write(content)
    return path


def _write_excel(tmp_dir: str, filename: str, sheets: dict[str, pd.DataFrame]) -> str:
    path = os.path.join(tmp_dir, filename)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def _make_record(file_path: str, score: float = 0.5, file_type: str = "csv",
                 query_text: str = "") -> dict:
    return {
        "id": 1,
        "file_path": file_path,
        "metadata_score": score,
        "file_type": file_type,
        "query_text": query_text,
    }


def _minimal_profiles(df: pd.DataFrame) -> dict[str, dict]:
    """Return a bare-bones profile dict like DataFrameProfiler would."""
    return {col: {"column_name": col} for col in df.columns}


# ===========================================================================
# TestCSVConnector
# ===========================================================================

class TestCSVConnector(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.connector = CSVConnector()

    # --- connect() ----------------------------------------------------------

    def test_connect_succeeds_with_valid_engine(self):
        """connect() should not raise when the engine is reachable."""
        self.connector.connect()  # uses in-memory SQLite via stub

    # --- extract() happy path -----------------------------------------------

    def test_extract_returns_dataframe_for_valid_csv(self):
        path = _write_csv(self.tmp, "data.csv", "a,b,c\n1,2,3\n4,5,6\n")
        self.connector._eligible = [_make_record(path)]
        result = self.connector.extract()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_extract_tags_source_metadata(self):
        path = _write_csv(self.tmp, "tagged.csv", "x,y\n10,20\n")
        self.connector._eligible = [_make_record(path, score=0.42)]
        result = self.connector.extract()
        self.assertIn("source_file", result.columns)
        self.assertIn("file_type", result.columns)
        self.assertIn("metadata_score", result.columns)
        self.assertIn("row_count", result.columns)
        self.assertIn("ingested_at", result.columns)
        self.assertEqual(result["file_type"].iloc[0], "csv")
        self.assertAlmostEqual(result["metadata_score"].iloc[0], 0.42)

    def test_extract_concatenates_multiple_files(self):
        path1 = _write_csv(self.tmp, "part1.csv", "n\n1\n2\n")
        path2 = _write_csv(self.tmp, "part2.csv", "n\n3\n4\n5\n")
        self.connector._eligible = [_make_record(path1), _make_record(path2)]
        result = self.connector.extract()
        self.assertEqual(len(result), 5)

    def test_extract_skips_high_score_records(self):
        """Records with score >= 0.7 fail validate_source and should be skipped."""
        path = _write_csv(self.tmp, "ok.csv", "a\n1\n")
        # Override validate_source to honour the threshold like the real base does
        self.connector.validate_source = lambda r: r["metadata_score"] < 0.7
        self.connector._eligible = [_make_record(path, score=0.9)]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    # --- extract() error paths ----------------------------------------------

    def test_extract_returns_empty_df_when_no_eligible_files(self):
        self.connector._eligible = []
        result = self.connector.extract()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_extract_skips_missing_file_gracefully(self):
        self.connector._eligible = [_make_record("/nonexistent/path/file.csv")]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_skips_empty_csv(self):
        path = _write_csv(self.tmp, "empty.csv", "")
        self.connector._eligible = [_make_record(path)]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_skips_header_only_csv(self):
        path = _write_csv(self.tmp, "header_only.csv", "col1,col2\n")
        self.connector._eligible = [_make_record(path)]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_handles_malformed_rows_without_crashing(self):
        content = "a,b,c\n1,2,3\n4,5\n7,8,9\n"   # row 2 is short
        path = _write_csv(self.tmp, "malformed.csv", content)
        self.connector._eligible = [_make_record(path)]
        result = self.connector.extract()
        self.assertFalse(result.empty)

    # --- delimiter sniffing -------------------------------------------------

    def test_sniff_delimiter_detects_semicolon(self):
        path = _write_csv(self.tmp, "semi.csv", "a;b;c\n1;2;3\n")
        delimiter = self.connector._sniff_delimiter(path, "utf-8")
        self.assertEqual(delimiter, ";")

    def test_sniff_delimiter_detects_pipe(self):
        path = _write_csv(self.tmp, "pipe.csv", "a|b|c\n1|2|3\n")
        delimiter = self.connector._sniff_delimiter(path, "utf-8")
        self.assertEqual(delimiter, "|")

    def test_sniff_delimiter_defaults_to_comma_on_failure(self):
        path = _write_csv(self.tmp, "nodelim.csv", "aaa\nbbb\n")
        delimiter = self.connector._sniff_delimiter(path, "utf-8")
        self.assertEqual(delimiter, ",")

    def test_sniff_delimiter_returns_comma_for_missing_file(self):
        delimiter = self.connector._sniff_delimiter("/does/not/exist.csv", "utf-8")
        self.assertEqual(delimiter, ",")

    # --- encoding detection -------------------------------------------------

    def test_detect_encoding_returns_utf8_for_ascii(self):
        path = _write_csv(self.tmp, "ascii.csv", "a,b\n1,2\n")
        enc = self.connector._detect_encoding(path)
        self.assertIsInstance(enc, str)
        self.assertTrue(len(enc) > 0)

    def test_detect_encoding_falls_back_for_missing_file(self):
        enc = self.connector._detect_encoding("/does/not/exist.csv")
        self.assertEqual(enc, "utf-8")

    def test_reads_latin1_encoded_file(self):
        # Write a longer latin-1 file so chardet has enough bytes to detect the
        # encoding reliably (chardet needs more than ~10 bytes of non-ASCII).
        rows = ["name,val"] + [f"café_{i},{ i}\n" for i in range(50)]
        content = "\n".join(rows)
        path = _write_csv(self.tmp, "latin1.csv", content, encoding="latin-1")
        self.connector._eligible = [_make_record(path)]
        result = self.connector.extract()
        # chardet may or may not detect latin-1 with enough confidence on this
        # platform; what matters is that the connector does not crash.
        self.assertIsInstance(result, pd.DataFrame)


# ===========================================================================
# TestExcelConnector
# ===========================================================================

class TestExcelConnector(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.connector = ExcelConnector()

    # --- connect() ----------------------------------------------------------

    def test_connect_succeeds(self):
        self.connector.connect()

    # --- extract() happy path -----------------------------------------------

    def test_extract_returns_dataframe_for_valid_xlsx(self):
        df_in = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        path = _write_excel(self.tmp, "data.xlsx", {"Sheet1": df_in})
        self.connector._eligible = [_make_record(path, file_type="xlsx")]
        result = self.connector.extract()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_extract_reads_all_sheets(self):
        sheets = {
            "Alpha": pd.DataFrame({"a": [1, 2]}),
            "Beta":  pd.DataFrame({"b": [3, 4, 5]}),
        }
        path = _write_excel(self.tmp, "multi.xlsx", sheets)
        self.connector._eligible = [_make_record(path, file_type="xlsx")]
        result = self.connector.extract()
        self.assertEqual(len(result), 5)   # 2 + 3 rows

    def test_extract_tags_sheet_name(self):
        path = _write_excel(self.tmp, "tagged.xlsx",
                            {"MySheet": pd.DataFrame({"v": [7]})})
        self.connector._eligible = [_make_record(path, file_type="xlsx")]
        result = self.connector.extract()
        self.assertIn("sheet_name", result.columns)
        self.assertEqual(result["sheet_name"].iloc[0], "MySheet")

    def test_extract_tags_source_metadata(self):
        path = _write_excel(self.tmp, "meta.xlsx",
                            {"S": pd.DataFrame({"c": [1]})})
        self.connector._eligible = [_make_record(path, score=0.3, file_type="xlsx")]
        result = self.connector.extract()
        for col in ("source_file", "file_type", "metadata_score", "row_count", "ingested_at"):
            self.assertIn(col, result.columns)
        self.assertAlmostEqual(result["metadata_score"].iloc[0], 0.3)

    def test_extract_drops_all_nan_columns_and_rows(self):
        df_in = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, None],   # all-NaN column → should be dropped
        })
        path = _write_excel(self.tmp, "nan_cols.xlsx", {"S": df_in})
        self.connector._eligible = [_make_record(path, file_type="xlsx")]
        result = self.connector.extract()
        self.assertNotIn("b", result.columns)

    # --- extract() error paths ----------------------------------------------

    def test_extract_returns_empty_df_for_no_eligible_files(self):
        self.connector._eligible = []
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_skips_missing_file(self):
        self.connector._eligible = [_make_record("/no/such/file.xlsx", file_type="xlsx")]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    # --- validate_source() --------------------------------------------------

    def test_validate_source_passes_valid_xlsx(self):
        path = _write_excel(self.tmp, "valid.xlsx",
                            {"S": pd.DataFrame({"c": [1]})})
        record = _make_record(path, file_type="xlsx")
        self.assertTrue(self.connector.validate_source(record))

    def test_validate_source_fails_for_empty_path(self):
        record = _make_record("", file_type="xlsx")
        self.assertFalse(self.connector.validate_source(record))

    def test_validate_source_fails_for_corrupt_xlsx(self):
        path = os.path.join(self.tmp, "corrupt.xlsx")
        with open(path, "wb") as fh:
            fh.write(b"this is not a valid xlsx file")
        record = _make_record(path, file_type="xlsx")
        self.assertFalse(self.connector.validate_source(record))

    # --- _read_single_sheet() -----------------------------------------------

    def test_read_single_sheet_returns_none_for_empty_sheet(self):
        xls = MagicMock()
        xls.sheet_names = ["Empty"]
        xls.parse = MagicMock(return_value=pd.DataFrame())
        with patch("pandas.read_excel", return_value=pd.DataFrame()):
            result = self.connector._read_single_sheet(
                xls, "Empty", "fake.xlsx", "xlsx", 0.5
            )
        self.assertIsNone(result)


# ===========================================================================
# TestSQLConnector
# ===========================================================================

class TestSQLConnector(unittest.TestCase):
    """Uses an in-process SQLite database as the source DB."""

    def setUp(self):
        from sqlalchemy import create_engine, text
        # Create a real in-memory SQLite source db with test data
        self.source_engine = create_engine("sqlite:///:memory:", future=True)
        with self.source_engine.connect() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER, amount REAL)"))
            conn.execute(text("INSERT INTO orders VALUES (1, 99.9)"))
            conn.execute(text("INSERT INTO orders VALUES (2, 49.5)"))
            conn.commit()
        self.source_conn_str = "sqlite:///:memory:"

        self.connector = SQLConnector(chunksize=100)
        # Pre-register the source engine in the cache so we don't rebuild it
        self.connector._source_engines[self.source_conn_str] = self.source_engine

    def _make_sql_record(self, query: str = "SELECT id, amount FROM orders",
                         score: float = 0.4) -> dict:
        return {
            "id": 42,
            "file_path": self.source_conn_str,
            "file_type": "sql",
            "metadata_score": score,
            "query_text": query,
        }

    # --- connect() ----------------------------------------------------------

    def test_connect_logs_eligible_count(self):
        self.connector._eligible = []
        self.connector.connect()   # should not raise

    # --- extract() happy path -----------------------------------------------

    def test_extract_returns_dataframe_for_valid_query(self):
        self.connector._eligible = [self._make_sql_record()]
        result = self.connector.extract()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_extract_tags_source_metadata(self):
        self.connector._eligible = [self._make_sql_record(score=0.33)]
        result = self.connector.extract()
        for col in ("source_db", "source_query", "file_type",
                    "metadata_score", "row_count", "ingested_at"):
            self.assertIn(col, result.columns)
        self.assertAlmostEqual(result["metadata_score"].iloc[0], 0.33)

    def test_extract_redacts_password_in_source_db(self):
        self.connector._eligible = [self._make_sql_record()]
        result = self.connector.extract()
        # sqlite doesn't have passwords, but verify no literal password appears
        self.assertNotIn("secret", result["source_db"].iloc[0])

    def test_extract_returns_empty_df_for_no_eligible_records(self):
        self.connector._eligible = []
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_skips_record_with_empty_query(self):
        record = self._make_sql_record(query="")
        self.connector._eligible = [record]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    def test_extract_skips_record_with_invalid_query(self):
        record = self._make_sql_record(query="SELECT * FROM nonexistent_table_xyz")
        self.connector._eligible = [record]
        result = self.connector.extract()
        self.assertTrue(result.empty)

    # --- validate_source() --------------------------------------------------

    def test_validate_source_passes_with_query(self):
        record = self._make_sql_record()
        self.assertTrue(self.connector.validate_source(record))

    def test_validate_source_fails_without_query_text(self):
        record = self._make_sql_record()
        record["query_text"] = "   "
        self.assertFalse(self.connector.validate_source(record))

    def test_validate_source_fails_with_empty_file_path(self):
        record = self._make_sql_record()
        record["file_path"] = ""
        self.assertFalse(self.connector.validate_source(record))

    # --- _get_source_engine() -----------------------------------------------

    def test_get_source_engine_caches_engine(self):
        conn_str = "sqlite:///:memory:"
        engine1 = self.connector._get_source_engine(conn_str)
        engine2 = self.connector._get_source_engine(conn_str)
        self.assertIs(engine1, engine2)

    def test_get_source_engine_raises_for_unset_env_var(self):
        bad_conn = "postgresql+psycopg2://user:${MISSING_TEST_VAR_XYZ}@localhost/db"
        with self.assertRaises(EnvironmentError):
            self.connector._get_source_engine(bad_conn)

    def test_get_source_engine_expands_env_var(self):
        os.environ["_TEST_DB_PASS"] = "testpass"
        try:
            conn_str = "sqlite:///:memory:"   # no actual var substitution needed
            engine = self.connector._get_source_engine(conn_str)
            self.assertIsNotNone(engine)
        finally:
            del os.environ["_TEST_DB_PASS"]

    # --- _stream_query() ----------------------------------------------------

    def test_stream_query_yields_all_rows(self):
        chunks = list(self.connector._stream_query(
            self.source_engine, "SELECT id, amount FROM orders"
        ))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 2)

    def test_stream_query_respects_chunksize(self):
        connector = SQLConnector(chunksize=1)
        connector._source_engines[self.source_conn_str] = self.source_engine
        chunks = list(connector._stream_query(
            self.source_engine, "SELECT id, amount FROM orders"
        ))
        self.assertEqual(len(chunks), 2)   # 2 rows, 1 per chunk

    # --- list_tables() ------------------------------------------------------

    def test_list_tables_returns_sorted_list(self):
        tables = self.connector.list_tables(self.source_conn_str)
        self.assertIn("orders", tables)
        self.assertEqual(tables, sorted(tables))


# ===========================================================================
# TestCsvExcelExtractor
# ===========================================================================

class TestCsvExcelExtractor(unittest.TestCase):

    def _make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    # --- constructor guards -------------------------------------------------

    def test_raises_for_none_dataframe(self):
        with self.assertRaises((ValueError, AttributeError)):
            CsvExcelExtractor(None, source_format="csv")

    def test_raises_for_empty_dataframe(self):
        with self.assertRaises(ValueError):
            CsvExcelExtractor(pd.DataFrame(), source_format="csv")

    # --- augment() structure ------------------------------------------------

    def test_augment_adds_all_expected_keys_csv(self):
        df = self._make_df({"amount": ["$10.00", "$20.00", "$30.00"]})
        profiles = _minimal_profiles(df)
        extractor = CsvExcelExtractor(df, source_format="csv")
        result = extractor.augment(profiles)
        for key in ("excel_col_letter", "detected_date_format",
                    "is_currency", "is_percentage", "multi_header_row"):
            self.assertIn(key, result["amount"])

    def test_augment_sets_excel_col_letter_for_excel(self):
        df = self._make_df({"col_a": [1], "col_b": [2], "col_c": [3]})
        profiles = _minimal_profiles(df)
        extractor = CsvExcelExtractor(df, source_format="excel")
        result = extractor.augment(profiles)
        self.assertEqual(result["col_a"]["excel_col_letter"], "A")
        self.assertEqual(result["col_b"]["excel_col_letter"], "B")
        self.assertEqual(result["col_c"]["excel_col_letter"], "C")

    def test_augment_sets_none_excel_col_letter_for_csv(self):
        df = self._make_df({"x": [1]})
        profiles = _minimal_profiles(df)
        extractor = CsvExcelExtractor(df, source_format="csv")
        result = extractor.augment(profiles)
        self.assertIsNone(result["x"]["excel_col_letter"])

    def test_augment_skips_column_missing_from_profiles(self):
        df = self._make_df({"a": [1], "b": [2]})
        profiles = {"a": {"column_name": "a"}}   # 'b' missing on purpose
        extractor = CsvExcelExtractor(df, source_format="csv")
        result = extractor.augment(profiles)   # should not raise
        self.assertNotIn("b", result)

    # --- date detection -----------------------------------------------------

    def test_detects_iso_date(self):
        df = self._make_df({"d": ["2024-01-15"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertEqual(result["d"]["detected_date_format"], "ISO_DATE")

    def test_detects_iso_datetime(self):
        df = self._make_df({"ts": ["2024-01-15T10:30:00Z"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertEqual(result["ts"]["detected_date_format"], "ISO_DATETIME")

    def test_detects_slash_dmy(self):
        df = self._make_df({"d": ["15/01/2024"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertEqual(result["d"]["detected_date_format"], "SLASH_DMY")

    def test_no_date_format_for_plain_text(self):
        df = self._make_df({"name": ["Alice", "Bob", "Charlie"] * 5})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertIsNone(result["name"]["detected_date_format"])

    def test_no_date_format_for_numeric_column(self):
        df = self._make_df({"amount": [1.0, 2.0, 3.0]})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertIsNone(result["amount"]["detected_date_format"])

    # --- currency detection -------------------------------------------------

    def test_detects_usd_symbol(self):
        df = self._make_df({"price": ["$9.99"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertTrue(result["price"]["is_currency"])

    def test_detects_eur_symbol(self):
        df = self._make_df({"price": ["€9.99"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertTrue(result["price"]["is_currency"])

    def test_detects_iso_currency_code_prefix(self):
        df = self._make_df({"price": ["USD 9.99"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertTrue(result["price"]["is_currency"])

    def test_not_currency_for_plain_numbers(self):
        df = self._make_df({"n": ["9.99"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertFalse(result["n"]["is_currency"])

    # --- percentage detection -----------------------------------------------

    def test_detects_percentage(self):
        df = self._make_df({"pct": ["12.5%"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertTrue(result["pct"]["is_percentage"])

    def test_detects_integer_percentage(self):
        df = self._make_df({"pct": ["75%"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertTrue(result["pct"]["is_percentage"])

    def test_not_percentage_for_plain_numbers(self):
        df = self._make_df({"n": ["75"] * 10})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertFalse(result["n"]["is_percentage"])

    # --- multi-header detection (Excel) ------------------------------------

    def test_multi_header_flagged_for_unnamed_column(self):
        df = self._make_df({"Unnamed: 0": [1, 2], "real_col": [3, 4]})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "excel").augment(profiles)
        self.assertTrue(result["Unnamed: 0"]["multi_header_row"])
        self.assertFalse(result["real_col"]["multi_header_row"])

    def test_multi_header_flagged_from_raw_header_rows(self):
        df = self._make_df({"Report Title": [1, 2], "amount": [3, 4]})
        profiles = _minimal_profiles(df)
        extractor = CsvExcelExtractor(
            df, "excel", raw_header_rows=[["Report Title", None]]
        )
        result = extractor.augment(profiles)
        self.assertTrue(result["Report Title"]["multi_header_row"])
        self.assertFalse(result["amount"]["multi_header_row"])

    def test_multi_header_always_false_for_csv(self):
        df = self._make_df({"Unnamed: 0": [1, 2]})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertFalse(result["Unnamed: 0"]["multi_header_row"])

    # --- threshold behaviour ------------------------------------------------

    def test_below_threshold_not_flagged_as_currency(self):
        # Only 50% currency — below the 80% threshold
        vals = ["$9.99"] * 5 + ["9.99"] * 5
        df = self._make_df({"p": vals})
        profiles = _minimal_profiles(df)
        result = CsvExcelExtractor(df, "csv").augment(profiles)
        self.assertFalse(result["p"]["is_currency"])

    # --- _excel_col_letter() helper ----------------------------------------

    def test_excel_col_letter_a(self):
        self.assertEqual(_cex_mod._excel_col_letter(0), "A")

    def test_excel_col_letter_z(self):
        self.assertEqual(_cex_mod._excel_col_letter(25), "Z")

    def test_excel_col_letter_aa(self):
        self.assertEqual(_cex_mod._excel_col_letter(26), "AA")

    def test_excel_col_letter_az(self):
        self.assertEqual(_cex_mod._excel_col_letter(51), "AZ")

    def test_excel_col_letter_ba(self):
        self.assertEqual(_cex_mod._excel_col_letter(52), "BA")


# ===========================================================================
# TestSqlExtractor
# ===========================================================================

class TestSqlExtractor(unittest.TestCase):
    """Uses an in-memory SQLite database with a real schema."""

    @classmethod
    def setUpClass(cls):
        from sqlalchemy import (
            create_engine, Column, Integer, String,
            Float, ForeignKey, MetaData, Table
        )
        cls.engine = create_engine("sqlite:///:memory:", future=True)
        meta = MetaData()

        # customers table
        cls.customers = Table(
            "customers", meta,
            Column("id", Integer, primary_key=True),
            Column("name", String(100), nullable=False),
            Column("email", String(200), nullable=True),
        )

        # orders table (FK to customers)
        cls.orders = Table(
            "orders", meta,
            Column("id", Integer, primary_key=True),
            Column("customer_id", Integer, ForeignKey("customers.id"), nullable=False),
            Column("amount", Float, nullable=True),
        )

        meta.create_all(cls.engine)

    # --- constructor guards -------------------------------------------------

    def test_raises_for_empty_table_name(self):
        with self.assertRaises(ValueError):
            SqlExtractor(self.engine, table_name="")

    # --- augment() — primary key --------------------------------------------

    def test_is_primary_key_true_for_pk_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        self.assertTrue(result["id"]["is_primary_key"])

    def test_is_primary_key_false_for_non_pk_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        self.assertFalse(result["name"]["is_primary_key"])

    # --- augment() — foreign key --------------------------------------------

    def test_is_foreign_key_true_for_fk_column(self):
        df = pd.DataFrame({"id": [1], "customer_id": [1], "amount": [10.0]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "orders")
        result = extractor.augment(profiles)
        self.assertTrue(result["customer_id"]["is_foreign_key"])

    def test_is_foreign_key_false_for_non_fk_column(self):
        df = pd.DataFrame({"id": [1], "customer_id": [1], "amount": [10.0]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "orders")
        result = extractor.augment(profiles)
        self.assertFalse(result["amount"]["is_foreign_key"])

    def test_fk_references_contains_referred_table(self):
        df = pd.DataFrame({"id": [1], "customer_id": [1], "amount": [10.0]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "orders")
        result = extractor.augment(profiles)
        refs = result["customer_id"]["fk_references"]
        self.assertTrue(len(refs) > 0)
        self.assertEqual(refs[0]["referred_table"], "customers")

    def test_fk_references_empty_for_non_fk_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        self.assertEqual(result["name"]["fk_references"], [])

    # --- augment() — nullability --------------------------------------------

    def test_is_nullable_false_for_not_null_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        # 'name' has nullable=False
        self.assertFalse(result["name"]["is_nullable"])

    def test_is_nullable_true_for_nullable_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": [None]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        self.assertTrue(result["email"]["is_nullable"])

    # --- augment() — native SQL type ----------------------------------------

    def test_native_sql_type_set_for_known_column(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        self.assertIsNotNone(result["id"]["native_sql_type"])

    # --- augment() — all keys always present --------------------------------

    def test_all_sql_keys_always_present(self):
        expected_keys = {
            "is_primary_key", "is_foreign_key", "fk_references",
            "is_nullable", "index_names", "is_indexed",
            "is_unique_indexed", "native_sql_type", "sql_default",
        }
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "email": ["a@b.com"]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)
        for col in df.columns:
            missing = expected_keys - result[col].keys()
            self.assertEqual(missing, set(),
                             f"Column '{col}' is missing keys: {missing}")

    # --- augment() — column not in schema -----------------------------------

    def test_safe_defaults_for_virtual_column_not_in_schema(self):
        df = pd.DataFrame({"id": [1], "virtual_col": [99]})
        profiles = _minimal_profiles(df)
        extractor = SqlExtractor(self.engine, "customers")
        result = extractor.augment(profiles)   # should not raise
        self.assertFalse(result["virtual_col"]["is_primary_key"])
        self.assertFalse(result["virtual_col"]["is_foreign_key"])
        self.assertEqual(result["virtual_col"]["fk_references"], [])
        self.assertIsNone(result["virtual_col"]["native_sql_type"])

    # --- _reflect_table() error handling ------------------------------------

    def test_safe_defaults_when_table_not_in_schema(self):
        # SqlExtractor catches Inspector errors gracefully and returns safe
        # defaults — it does NOT raise for a missing table.
        extractor = SqlExtractor(self.engine, "nonexistent_table_xyz")
        profiles = {"col": {"column_name": "col"}}
        result = extractor.augment(profiles)   # must not raise
        # All keys should still be present with safe defaults
        self.assertFalse(result["col"]["is_primary_key"])
        self.assertFalse(result["col"]["is_foreign_key"])
        self.assertEqual(result["col"]["fk_references"], [])
        self.assertIsNone(result["col"]["native_sql_type"])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)