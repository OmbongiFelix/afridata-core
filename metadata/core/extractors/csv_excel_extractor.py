"""
Structural metadata extractor for CSV and Excel sources.

Operates on a pd.DataFrame after ingestion and augments the
ColumnProfile with source-format-specific features:

    - Detects multi-header rows in Excel (e.g. merged title rows)
    - Identifies date/time columns using format pattern matching
    - Flags currency and percentage columns from string patterns
    - Extracts original Excel column letters for cross-reference

Works alongside ColumnProfiler — call profiler first, then pass
profiles into this extractor for augmentation.

Usage:
    from profiler import DataFrameProfiler
    from csv_excel_extractor import CsvExcelExtractor

    profiles = DataFrameProfiler(df).run()
    extractor = CsvExcelExtractor(df, source_format="excel")
    augmented = extractor.augment(profiles)
"""

import logging
import re
import string
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Date / time format fingerprints
# Ordered from most-specific to least-specific to avoid false positives.
# ---------------------------------------------------------------------------
_DATETIME_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ISO-8601 datetime with optional fractional seconds and timezone
    ("ISO_DATETIME",    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")),
    # ISO date only
    ("ISO_DATE",        re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    # Common locale formats: dd/mm/yyyy or mm/dd/yyyy
    ("SLASH_DMY",       re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    # dd-Mon-yyyy  e.g. 01-Jan-2024
    ("DMY_MON",         re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")),
    # Mon dd, yyyy  e.g. Jan 01, 2024
    ("MON_LONG",        re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$")),
    # yyyy/mm/dd
    ("YMD_SLASH",       re.compile(r"^\d{4}/\d{2}/\d{2}$")),
    # Time only: HH:MM or HH:MM:SS
    ("TIME_ONLY",       re.compile(r"^\d{2}:\d{2}(:\d{2})?$")),
]

# ---------------------------------------------------------------------------
# Currency patterns
# Recognises leading/trailing symbols and common ISO codes.
# ---------------------------------------------------------------------------
_CURRENCY_SYMBOL_RE = re.compile(
    r"""
    (
        ^\s*[$€£¥₹₩₪₦₫฿₴₽¢]   # leading symbol
        |
        [$€£¥₹₩₪₦₫฿₴₽¢]\s*$   # trailing symbol
        |
        ^\s*(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|INR|KES|ZAR)\s  # ISO code prefix
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Percentage patterns
# ---------------------------------------------------------------------------
_PERCENTAGE_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*%\s*$")

# ---------------------------------------------------------------------------
# Minimum proportion of non-null values that must match a pattern for the
# column to be labelled with that format.
# ---------------------------------------------------------------------------
FORMAT_MATCH_THRESHOLD = 0.80

# Maximum number of values sampled per column for pattern-matching.
# Capped to keep augmentation fast on wide, long files.
PATTERN_SAMPLE_CAP = 500


def _excel_col_letter(zero_based_index: int) -> str:
    """
    Convert a zero-based column index to an Excel column letter string.

    Examples:
        0  -> 'A'
        25 -> 'Z'
        26 -> 'AA'
        701 -> 'ZZ'

    Args:
        zero_based_index: Non-negative integer column position.

    Returns:
        Upper-case Excel column letter string (e.g. 'A', 'BC').
    """
    result = []
    n = zero_based_index
    while True:
        n, remainder = divmod(n, 26)
        result.append(string.ascii_uppercase[remainder])
        if n == 0:
            break
        n -= 1  # Excel columns are 1-indexed within each "digit"
    return "".join(reversed(result))


class CsvExcelExtractor:
    """
    Augments profiler output with CSV/Excel-specific structural metadata.

    For each column the extractor may add the following keys to the profile
    dict (all keys are always present after augmentation, defaulting to None
    or False so downstream consumers need no defensive .get() calls):

        excel_col_letter  (str | None)
            Excel column letter (e.g. 'A', 'BC'). Set for Excel sources;
            None for CSV sources.

        detected_date_format  (str | None)
            Date/time format label from _DATETIME_PATTERNS, or None if the
            column does not match any recognised date/time pattern.

        is_currency  (bool)
            True when ≥FORMAT_MATCH_THRESHOLD of non-null string values
            carry a recognisable currency symbol or ISO code prefix.

        is_percentage  (bool)
            True when ≥FORMAT_MATCH_THRESHOLD of non-null string values
            end with '%'.

        multi_header_row  (bool)
            True when the column header appears to be a merged/title row
            rather than a genuine field name (Excel-specific heuristic).

    Usage:
        profiles = DataFrameProfiler(df).run()
        extractor = CsvExcelExtractor(df, source_format="excel")
        augmented = extractor.augment(profiles)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source_format: str = "csv",
        raw_header_rows: list[list[Any]] | None = None,
    ):
        """
        Args:
            df:
                The ingested DataFrame (post-profiling).
            source_format:
                Either ``"csv"`` or ``"excel"`` (case-insensitive).
                Controls whether Excel-specific features (column letters,
                multi-header detection) are computed.
            raw_header_rows:
                Optional list of raw rows read *before* the actual header
                row in the source file (e.g. a title banner row in Excel).
                Used exclusively for multi-header detection.  When None the
                extractor skips that heuristic.
        """
        if df is None or df.empty:
            raise ValueError(
                "CsvExcelExtractor received an empty DataFrame. "
                "Ensure the connector's extract() returned data before augmenting."
            )
        self.df = df
        self.source_format = source_format.lower()
        self.raw_header_rows = raw_header_rows or []
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def augment(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Augment an existing profile dict produced by DataFrameProfiler.

        The method operates in-place on each inner dict *and* returns the
        mutated mapping so callers can chain calls.

        Args:
            profiles:
                Output of ``DataFrameProfiler.run()`` — a dict keyed by
                column name, each value being a ``ColumnProfile.to_dict()``
                result.

        Returns:
            The same ``profiles`` dict with extra keys injected per column.
        """
        is_excel = self.source_format == "excel"
        multi_header_cols = self._detect_multi_header_columns() if is_excel else set()

        self.logger.info(
            "CsvExcelExtractor.augment: augmenting %d column(s) [format=%s].",
            len(profiles),
            self.source_format,
        )

        for idx, col in enumerate(self.df.columns):
            if col not in profiles:
                self.logger.warning(
                    "CsvExcelExtractor: column '%s' present in DataFrame but missing "
                    "from profiles — skipping augmentation for this column.",
                    col,
                )
                continue

            profile = profiles[col]

            # Excel column letter (None for CSV)
            profile["excel_col_letter"] = _excel_col_letter(idx) if is_excel else None

            # Multi-header heuristic (Excel only)
            profile["multi_header_row"] = col in multi_header_cols

            # Pattern-based format detection on the raw series
            series = self.df[col]
            date_fmt, is_currency, is_pct = self._analyse_string_patterns(series)
            profile["detected_date_format"] = date_fmt
            profile["is_currency"] = is_currency
            profile["is_percentage"] = is_pct

            self.logger.debug(
                "Augmented column '%s': excel_col=%s date_fmt=%s currency=%s pct=%s",
                col,
                profile["excel_col_letter"],
                date_fmt,
                is_currency,
                is_pct,
            )

        self.logger.info(
            "CsvExcelExtractor.augment: completed augmentation for %d column(s).",
            len(profiles),
        )
        return profiles

    # ------------------------------------------------------------------
    # Multi-header detection
    # ------------------------------------------------------------------

    def _detect_multi_header_columns(self) -> set[str]:
        """
        Identify DataFrame columns whose header looks like a merged title
        row rather than a genuine field name.

        Two signals are combined:

        1. **Raw header rows**: If ``raw_header_rows`` were supplied, any
           column name that appears verbatim in those rows is flagged.

        2. **Unnamed column heuristic**: pandas names unnamed merged cells
           ``"Unnamed: N_level_M"`` when reading with ``header=[0, 1]`` etc.
           Such columns are always flagged.

        Returns:
            Set of column name strings that appear to be title/merged rows.
        """
        flagged: set[str] = set()

        # Heuristic 1 – values from raw pre-header rows
        raw_values: set[str] = set()
        for row in self.raw_header_rows:
            for cell in row:
                if cell is not None:
                    raw_values.add(str(cell).strip())

        # Heuristic 2 – pandas "Unnamed" columns from merged-cell reads
        _unnamed_re = re.compile(r"^Unnamed:\s*\d+(_level_\d+)?$", re.IGNORECASE)

        for col in self.df.columns:
            col_str = str(col).strip()
            if col_str in raw_values:
                flagged.add(col)
            elif _unnamed_re.match(col_str):
                flagged.add(col)

        if flagged:
            self.logger.debug(
                "CsvExcelExtractor: %d column(s) flagged as multi-header: %s",
                len(flagged),
                flagged,
            )

        return flagged

    # ------------------------------------------------------------------
    # String pattern analysis
    # ------------------------------------------------------------------

    def _analyse_string_patterns(
        self, series: pd.Series
    ) -> tuple[str | None, bool, bool]:
        """
        Scan a column's non-null string values and return format labels.

        Only object/string-dtype columns are analysed.  Numeric and
        datetime columns that pandas has already typed are passed through
        without pattern-matching (their typing is already correct).

        Sampling is capped at PATTERN_SAMPLE_CAP to keep large DataFrames
        fast.

        Args:
            series: Raw column from the DataFrame.

        Returns:
            A 3-tuple of:
                detected_date_format (str | None)
                is_currency          (bool)
                is_percentage        (bool)
        """
        # Skip pattern analysis for columns pandas already typed correctly.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            return None, False, False

        clean = series.dropna().astype(str)
        if clean.empty:
            return None, False, False

        # Cap sample size for performance.
        sample = clean.iloc[:PATTERN_SAMPLE_CAP]
        n = len(sample)

        detected_date_fmt = self._detect_date_format(sample, n)
        is_currency = self._detect_currency(sample, n)
        is_percentage = self._detect_percentage(sample, n)

        return detected_date_fmt, is_currency, is_percentage

    def _detect_date_format(self, sample: pd.Series, n: int) -> str | None:
        """
        Return the label of the first date/time pattern that matches
        ≥FORMAT_MATCH_THRESHOLD of values in ``sample``, or None.

        Patterns are evaluated in declaration order (most-specific first).

        Args:
            sample: Non-null string values to test (already cast to str).
            n:      Total count of values in the sample (pre-computed).

        Returns:
            Pattern label string (e.g. ``"ISO_DATE"``) or None.
        """
        for label, pattern in _DATETIME_PATTERNS:
            match_count = sample.str.match(pattern).sum()
            if match_count / n >= FORMAT_MATCH_THRESHOLD:
                self.logger.debug(
                    "_detect_date_format: matched '%s' with %.0f%% coverage.",
                    label,
                    match_count / n * 100,
                )
                return label
        return None

    def _detect_currency(self, sample: pd.Series, n: int) -> bool:
        """
        Return True when ≥FORMAT_MATCH_THRESHOLD of ``sample`` values
        carry a recognisable currency marker.

        Args:
            sample: Non-null string values.
            n:      Total count of values in the sample.

        Returns:
            bool
        """
        match_count = sample.str.contains(_CURRENCY_SYMBOL_RE, regex=True).sum()
        return (match_count / n) >= FORMAT_MATCH_THRESHOLD

    def _detect_percentage(self, sample: pd.Series, n: int) -> bool:
        """
        Return True when ≥FORMAT_MATCH_THRESHOLD of ``sample`` values
        match a numeric-percentage pattern (e.g. ``"12.5%"``).

        Args:
            sample: Non-null string values.
            n:      Total count of values in the sample.

        Returns:
            bool
        """
        match_count = sample.str.match(_PERCENTAGE_RE).sum()
        return (match_count / n) >= FORMAT_MATCH_THRESHOLD