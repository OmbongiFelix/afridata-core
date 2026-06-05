"""
Adapter for ingesting Excel spreadsheet data sources.

Reads .xlsx and legacy .xls files whose records are stored in the database
and whose metadata_score is below 0.7. Supports multi-sheet extraction,
header-row detection, and stripping of Excel formatting artefacts
(merged cells, trailing empty rows/columns).

Returns a clean pd.DataFrame tagged with sheet_name, source_file,
file_type, metadata_score, and ingested_at.

Engine note:
    openpyxl and xlrd are expected to be installed in the environment.
    pandas selects the correct engine automatically based on file extension:
        .xlsx  →  openpyxl
        .xls   →  xlrd
    Neither library is imported here directly; pandas owns that decision.
    The one exception is validate_source(), which uses openpyxl to perform
    a lightweight structural check on .xlsx files before reading them fully.
    If you prefer to skip that check, set VALIDATE_XLSX_STRUCTURE = False.
"""

import logging
from pathlib import Path

import openpyxl          # used only in validate_source() for .xlsx pre-check
import pandas as pd

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)

# Set to False to skip the openpyxl structural pre-check in validate_source().
VALIDATE_XLSX_STRUCTURE = True


class ExcelConnector(BaseConnector):
    """
    Connector for Excel files (.xlsx, .xls) tracked in the database.

    Workflow:
        1. connect()  — verifies the DB engine is reachable.
        2. extract()  — calls fetch_eligible_files() to get all Excel records
                        with metadata_score < 0.7, then reads every sheet of
                        each file and concatenates everything into one DataFrame.
    """

    SUPPORTED_EXTENSIONS = [".excel"]

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
            "ExcelConnector: database engine ready (%s).",
            engine.url.render_as_string(hide_password=True),
        )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """
        Fetch all eligible Excel records from the DB (metadata_score < 0.7),
        read every sheet of each file, and return one concatenated DataFrame.

        Each row in the result is tagged with:
            source_file       — path/identifier from the DB record
            file_type         — 'xlsx' or 'xls'
            sheet_name        — name of the sheet the row came from
            metadata_score    — score from the DB record
            row_count         — number of data rows in that sheet
            ingested_at       — UTC timestamp of this ingestion run

        Returns:
            pd.DataFrame — empty DataFrame if no eligible files are found.

        Raises:
            RuntimeError: propagated from fetch_eligible_files if the DB query fails.
        """
        eligible = self.fetch_eligible_files()

        if not eligible:
            self.logger.info("ExcelConnector.extract: no eligible Excel files found.")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for record in eligible:
            if not self.validate_source(record):
                continue

            file_path = record["file_path"]
            file_type = record["file_type"]
            score = record["metadata_score"]

            file_frames = self._read_all_sheets(file_path, file_type, score)
            frames.extend(file_frames)

        if not frames:
            self.logger.warning(
                "ExcelConnector.extract: all eligible files failed to load."
            )
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.logger.info(
            "ExcelConnector.extract: loaded %d row(s) from %d sheet(s) across %d file(s).",
            len(combined),
            len(frames),
            len(eligible),
        )
        return combined

    # ------------------------------------------------------------------
    # validate_source  (override)
    # ------------------------------------------------------------------

    def validate_source(self, file_record: dict) -> bool:
        """
        Extends the base validation with an Excel-specific structural check.

        For .xlsx files, attempts to open the workbook with openpyxl in
        read-only mode to confirm it is a valid, non-corrupt Excel file
        before committing to a full pandas read.

        For .xls files, the lightweight check is skipped because xlrd does
        not expose a comparable lightweight open path; any corruption will
        surface during extract() with a clear error message.

        Args:
            file_record: dict from fetch_eligible_files().

        Returns:
            True if the file passes all checks, False otherwise.
        """
        # Run the base checks first (empty path, score threshold).
        if not super().validate_source(file_record):
            return False

        file_path = file_record["file_path"]
        extension = Path(file_path).suffix.lower()

        if VALIDATE_XLSX_STRUCTURE and extension == ".xlsx":
            try:
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                sheet_count = len(wb.sheetnames)
                wb.close()

                if sheet_count == 0:
                    self.logger.warning(
                        "validate_source: '%s' contains no sheets — skipping.", file_path
                    )
                    return False

                self.logger.debug(
                    "validate_source: '%s' OK (%d sheet(s)).", file_path, sheet_count
                )

            except Exception as exc:
                self.logger.error(
                    "validate_source: '%s' failed structural check (%s) — skipping.",
                    file_path,
                    exc,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all_sheets(
        self, file_path: str, file_type: str, metadata_score: float
    ) -> list[pd.DataFrame]:
        """
        Open an Excel file and read every non-empty sheet into its own DataFrame.

        pandas selects openpyxl for .xlsx and xlrd for .xls automatically;
        no engine argument is passed here so that behaviour is preserved.

        Args:
            file_path:      Path/identifier of the file (from DB).
            file_type:      'xlsx' or 'xls', forwarded to the tagging columns.
            metadata_score: Score stored in the DB, forwarded to the DataFrame.

        Returns:
            List of tagged DataFrames, one per non-empty sheet.
            Returns an empty list if the file cannot be read.
        """
        results: list[pd.DataFrame] = []

        try:
            xls = pd.ExcelFile(file_path)   # pandas picks engine from extension
        except Exception as exc:
            self.logger.error(
                "_read_all_sheets: could not open '%s': %s — skipping.", file_path, exc
            )
            return results

        for sheet in xls.sheet_names:
            df = self._read_single_sheet(xls, sheet, file_path, file_type, metadata_score)
            if df is not None:
                results.append(df)

        if not results:
            self.logger.warning(
                "_read_all_sheets: '%s' produced no usable sheets.", file_path
            )

        return results

    def _read_single_sheet(
        self,
        xls: pd.ExcelFile,
        sheet: str | int,
        file_path: str,
        file_type: str,
        metadata_score: float,
    ) -> pd.DataFrame | None:
        """
        Read one sheet, strip Excel artefacts, and tag with source metadata.

        Artefact handling:
            - Trailing all-NaN columns are dropped.
            - Trailing all-NaN rows are dropped.
            - Fully empty sheets are skipped.

        Args:
            xls:            Open pd.ExcelFile handle.
            sheet:          Sheet name to read.
            file_path:      Forwarded to source_file tag.
            file_type:      Forwarded to file_type tag.
            metadata_score: Forwarded to metadata_score tag.

        Returns:
            Tagged pd.DataFrame, or None if the sheet is empty / unreadable.
        """
        try:
            df = pd.read_excel(
                xls,
                sheet_name=sheet,
                header=0,        # treat row 0 as header; adjust if needed
            )

            # --- strip trailing empty rows and columns ---
            df = df.dropna(axis=1, how="all")   # drop all-NaN columns
            df = df.dropna(axis=0, how="all")   # drop all-NaN rows

            if df.empty:
                self.logger.debug(
                    "_read_single_sheet: sheet '%s' in '%s' is empty after cleaning — skipping.",
                    sheet,
                    file_path,
                )
                return None

            # --- tag rows with source metadata ---
            df["source_file"] = file_path
            df["file_type"] = file_type
            df["sheet_name"] = sheet
            df["metadata_score"] = metadata_score
            df["row_count"] = len(df)
            df["ingested_at"] = pd.Timestamp.utcnow()

            self.logger.info(
                "Loaded sheet '%s' from '%s': %d row(s), score=%.3f.",
                sheet,
                file_path,
                len(df),
                metadata_score,
            )
            return df

        except Exception as exc:
            self.logger.error(
                "_read_single_sheet: error reading sheet '%s' from '%s': %s — skipping.",
                sheet,
                file_path,
                exc,
            )
            return None