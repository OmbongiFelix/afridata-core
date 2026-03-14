"""
Adapter for ingesting CSV data sources.

Reads one or more CSV files whose records are stored in the database and whose
metadata_score is below 0.7. Handles encoding detection, delimiter sniffing,
and malformed-row recovery. Returns a clean pd.DataFrame tagged with source
metadata (source_file, file_type, metadata_score, row_count, ingested_at).

Supports files referenced in the DB registry only.
Local-only paths and remote HTTP URLs are intentionally excluded.
"""

import csv
import io
import logging

import chardet
import pandas as pd

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)


class CSVConnector(BaseConnector):
    """
    Connector for CSV files tracked in the database.

    Workflow:
        1. connect()  — verifies the DB engine is reachable.
        2. extract()  — calls fetch_eligible_files() to get all CSV records
                        with metadata_score < 0.7, then reads and concatenates
                        each one into a single DataFrame.
    """

    SUPPORTED_EXTENSIONS = [".csv"]

    # Delimiters tried during sniffing, in priority order.
    _CANDIDATE_DELIMITERS = [",", ";", "|", "\t"]

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify that a database engine can be created.
        Raises EnvironmentError / SQLAlchemyError on failure (from get_engine).
        """
        engine = self.get_engine()
        self.logger.info(
            "CSVConnector: database engine ready (%s).", engine.url.render_as_string(hide_password=True)
        )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """
        Fetch all eligible CSV records from the DB (metadata_score < 0.7),
        read each file, and return one concatenated DataFrame.

        Each row in the result is tagged with:
            source_file       — path/identifier from the DB record
            file_type         — 'csv'
            metadata_score    — score from the DB record
            row_count         — number of data rows in that file
            ingested_at       — UTC timestamp of this ingestion run

        Returns:
            pd.DataFrame — empty DataFrame if no eligible files are found.

        Raises:
            RuntimeError: propagated from fetch_eligible_files if the DB query fails.
        """
        eligible = self.fetch_eligible_files()

        if not eligible:
            self.logger.info("CSVConnector.extract: no eligible CSV files found.")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for record in eligible:
            if not self.validate_source(record):
                continue

            file_path = record["file_path"]
            score = record["metadata_score"]

            df = self._read_single_csv(file_path, score)
            if df is not None:
                frames.append(df)

        if not frames:
            self.logger.warning("CSVConnector.extract: all eligible files failed to load.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.logger.info(
            "CSVConnector.extract: loaded %d row(s) from %d file(s).",
            len(combined),
            len(frames),
        )
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_single_csv(self, file_path: str, metadata_score: float) -> pd.DataFrame | None:
        """
        Read one CSV file, auto-detecting encoding and delimiter.

        Args:
            file_path:      Path / identifier of the file (from DB).
            metadata_score: Score stored in the DB, forwarded to the DataFrame.

        Returns:
            Tagged pd.DataFrame, or None if the file cannot be read.
        """
        try:
            encoding = self._detect_encoding(file_path)
            delimiter = self._sniff_delimiter(file_path, encoding)

            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                on_bad_lines="warn",   # skip malformed rows, don't crash
                engine="python",       # needed for multi-char / sniffed delimiters
            )

            if df.empty:
                self.logger.warning("'%s' loaded but is empty — skipping.", file_path)
                return None

            # --- tag rows with source metadata ---
            df["source_file"] = file_path
            df["file_type"] = "csv"
            df["metadata_score"] = metadata_score
            df["row_count"] = len(df)
            df["ingested_at"] = pd.Timestamp.utcnow()

            self.logger.info(
                "Loaded '%s': %d row(s), encoding=%s, delimiter=%r, score=%.3f.",
                file_path,
                len(df),
                encoding,
                delimiter,
                metadata_score,
            )
            return df

        except FileNotFoundError:
            self.logger.error("File not found: '%s'. Skipping.", file_path)
        except pd.errors.EmptyDataError:
            self.logger.error("'%s' is empty or has no parseable content. Skipping.", file_path)
        except Exception as exc:
            self.logger.error("Unexpected error reading '%s': %s. Skipping.", file_path, exc)

        return None

    def _detect_encoding(self, file_path: str) -> str:
        """
        Read the first 10 KB of the file and use chardet to detect its encoding.
        Falls back to 'utf-8' if detection is inconclusive.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Encoding string, e.g. 'utf-8', 'latin-1'.
        """
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read(10_000)
            result = chardet.detect(raw)
            encoding = result.get("encoding") or "utf-8"
            confidence = result.get("confidence", 0.0)

            if confidence < 0.6:
                self.logger.debug(
                    "_detect_encoding: low confidence (%.2f) for '%s', defaulting to utf-8.",
                    confidence,
                    file_path,
                )
                return "utf-8"

            self.logger.debug(
                "_detect_encoding: '%s' → %s (confidence %.2f).",
                file_path,
                encoding,
                confidence,
            )
            return encoding

        except OSError as exc:
            self.logger.warning(
                "_detect_encoding: could not read '%s' (%s). Defaulting to utf-8.", file_path, exc
            )
            return "utf-8"

    def _sniff_delimiter(self, file_path: str, encoding: str) -> str:
        """
        Read the first 4 KB of the file and attempt to determine the delimiter
        using csv.Sniffer. Falls back to comma if sniffing fails.

        Args:
            file_path: Path to the CSV file.
            encoding:  Encoding to use when reading the sample.

        Returns:
            Single-character delimiter string.
        """
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as fh:
                sample = fh.read(4_096)

            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters="".join(self._CANDIDATE_DELIMITERS))
            self.logger.debug(
                "_sniff_delimiter: '%s' → delimiter %r.", file_path, dialect.delimiter
            )
            return dialect.delimiter

        except csv.Error:
            self.logger.debug(
                "_sniff_delimiter: could not detect delimiter for '%s'. Defaulting to ','.",
                file_path,
            )
            return ","
        except OSError as exc:
            self.logger.warning(
                "_sniff_delimiter: could not open '%s' (%s). Defaulting to ','.", file_path, exc
            )
            return ","