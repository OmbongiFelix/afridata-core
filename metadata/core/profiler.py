"""
Column profiler for the Metadata Extraction Pipeline.

Analyses a pd.DataFrame and produces a per-column profile dict
containing statistical summaries, data quality indicators, and
inferred primitive types. This profile feeds both the SemanticClassifier
and the LLMMetadataGenerator as structured context.

Profile fields per column:
    dtype         - pandas dtype string
    null_pct      - percentage of null / NaN values
    unique_count  - count of distinct values
    sample_values - up to 5 representative values
    min/max/mean  - numeric stats (if applicable)
    is_id_like    - heuristic flag for ID columns

"""

import logging
import re
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Number of representative sample values to capture per column.
SAMPLE_SIZE = 5

# If unique_count / total_rows exceeds this ratio and the column name
# contains an ID-like token, the column is flagged as is_id_like.
ID_UNIQUENESS_THRESHOLD = 0.95

# Regex patterns used by the is_id_like heuristic.
_ID_NAME_PATTERN = re.compile(
    r"(^|_)(id|key|uuid|guid|pk|ref|code|num|number|no)($|_)",
    re.IGNORECASE,
)


class ColumnProfile:
    """
    Holds the computed profile for a single DataFrame column.

    Attributes are set by DataFrameProfiler and consumed downstream
    by the SemanticClassifier and LLMMetadataGenerator.
    """

    __slots__ = (
        "column_name",
        "dtype",
        "row_count",
        "null_count",
        "null_pct",
        "unique_count",
        "unique_pct",
        "sample_values",
        "is_id_like",
        # numeric only
        "min",
        "max",
        "mean",
        "median",
        "std",
        # string only
        "min_length",
        "max_length",
        "mean_length",
        # datetime only
        "earliest",
        "latest",
    )

    def __init__(self, column_name: str):
        self.column_name = column_name
        self.dtype: str = ""
        self.row_count: int = 0
        self.null_count: int = 0
        self.null_pct: float = 0.0
        self.unique_count: int = 0
        self.unique_pct: float = 0.0
        self.sample_values: list[Any] = []
        self.is_id_like: bool = False
        # numeric
        self.min: float | None = None
        self.max: float | None = None
        self.mean: float | None = None
        self.median: float | None = None
        self.std: float | None = None
        # string
        self.min_length: int | None = None
        self.max_length: int | None = None
        self.mean_length: float | None = None
        # datetime
        self.earliest: str | None = None
        self.latest: str | None = None

    def to_dict(self) -> dict:
        """
        Serialise the profile to a plain dict suitable for passing to
        downstream stages (classifier, LLM generator, schema builder).

        None values are included so downstream consumers can rely on key
        presence without defensive .get() calls.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"<ColumnProfile column='{self.column_name}' dtype={self.dtype} "
            f"null_pct={self.null_pct:.1f}% unique={self.unique_count}>"
        )


class DataFrameProfiler:
    """
    Computes a ColumnProfile for every column in a pd.DataFrame.

    Usage:
        profiler = DataFrameProfiler(df)
        profiles = profiler.run()
        # profiles is a dict[str, dict] keyed by column name.

    The profiler is intentionally stateless between run() calls —
    pass a new DataFrame to get a fresh profile.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: The DataFrame produced by any connector's extract() method.
                Must be non-empty; raises ValueError otherwise.
        """
        if df is None or df.empty:
            raise ValueError(
                "DataFrameProfiler received an empty DataFrame. "
                "Ensure the connector's extract() returned data before profiling."
            )
        self.df = df
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict[str, dict]:
        """
        Profile every column in the DataFrame.

        Returns:
            dict keyed by column name, each value being the output of
            ColumnProfile.to_dict(). Example:

            {
                "customer_id": {
                    "column_name": "customer_id",
                    "dtype": "int64",
                    "null_pct": 0.0,
                    "unique_count": 10000,
                    "is_id_like": True,
                    "min": 1.0, "max": 10000.0, ...
                },
                "email": { ... },
                ...
            }
        """
        self.logger.info(
            "DataFrameProfiler.run: profiling %d column(s) across %d row(s).",
            len(self.df.columns),
            len(self.df),
        )

        profiles: dict[str, dict] = {}

        for col in self.df.columns:
            try:
                profile = self._profile_column(col)
                profiles[col] = profile.to_dict()
            except Exception as exc:
                self.logger.error(
                    "DataFrameProfiler: failed to profile column '%s': %s — skipping.",
                    col,
                    exc,
                )

        self.logger.info(
            "DataFrameProfiler.run: completed %d/%d column profile(s).",
            len(profiles),
            len(self.df.columns),
        )
        return profiles

    # ------------------------------------------------------------------
    # Per-column dispatch
    # ------------------------------------------------------------------

    def _profile_column(self, col: str) -> ColumnProfile:
        """
        Build a ColumnProfile for a single column, dispatching to
        type-specific helpers for numeric, datetime, and string columns.

        Args:
            col: Column name present in self.df.

        Returns:
            Populated ColumnProfile instance.
        """
        series = self.df[col]
        profile = ColumnProfile(col)
        total = len(series)

        # --- universal fields ---
        profile.dtype = str(series.dtype)
        profile.row_count = total
        profile.null_count = int(series.isna().sum())
        profile.null_pct = round(profile.null_count / total * 100, 2) if total else 0.0
        profile.unique_count = int(series.nunique(dropna=True))
        profile.unique_pct = round(profile.unique_count / total * 100, 2) if total else 0.0
        profile.sample_values = self._sample_values(series)
        profile.is_id_like = self._is_id_like(col, profile.unique_pct)

        # --- type-specific fields ---
        if pd.api.types.is_numeric_dtype(series):
            self._profile_numeric(series, profile)
        elif pd.api.types.is_datetime64_any_dtype(series):
            self._profile_datetime(series, profile)
        else:
            # Attempt datetime coercion before falling back to string profiling.
            coerced = self._try_coerce_datetime(series)
            if coerced is not None:
                profile.dtype = "datetime64[ns] (coerced)"
                self._profile_datetime(coerced, profile)
            else:
                self._profile_string(series, profile)

        self.logger.debug("Profiled column '%s': %r", col, profile)
        return profile

    # ------------------------------------------------------------------
    # Type-specific profilers
    # ------------------------------------------------------------------

    def _profile_numeric(self, series: pd.Series, profile: ColumnProfile) -> None:
        """
        Populate numeric stats: min, max, mean, median, std.
        Skips gracefully if the non-null slice is empty.
        """
        clean = series.dropna()
        if clean.empty:
            return

        profile.min = self._safe_scalar(clean.min())
        profile.max = self._safe_scalar(clean.max())
        profile.mean = round(float(clean.mean()), 6)
        profile.median = self._safe_scalar(clean.median())
        profile.std = round(float(clean.std()), 6) if len(clean) > 1 else 0.0

    def _profile_datetime(self, series: pd.Series, profile: ColumnProfile) -> None:
        """
        Populate datetime stats: earliest and latest as ISO-8601 strings.
        Skips gracefully if the non-null slice is empty.
        """
        clean = series.dropna()
        if clean.empty:
            return

        profile.earliest = pd.Timestamp(clean.min()).isoformat()
        profile.latest = pd.Timestamp(clean.max()).isoformat()

    def _profile_string(self, series: pd.Series, profile: ColumnProfile) -> None:
        """
        Populate string stats: min_length, max_length, mean_length.
        Converts values to str before measuring to handle mixed-type columns.
        Skips gracefully if the non-null slice is empty.
        """
        clean = series.dropna()
        if clean.empty:
            return

        lengths = clean.astype(str).str.len()
        profile.min_length = int(lengths.min())
        profile.max_length = int(lengths.max())
        profile.mean_length = round(float(lengths.mean()), 2)

    # ------------------------------------------------------------------
    # Heuristics and utilities
    # ------------------------------------------------------------------

    def _is_id_like(self, col_name: str, unique_pct: float) -> bool:
        """
        Flag a column as ID-like when both conditions are true:
            1. The column name matches a known ID-token pattern.
            2. The uniqueness ratio exceeds ID_UNIQUENESS_THRESHOLD.

        This is a heuristic — the SemanticClassifier may override it.

        Args:
            col_name:   Column name string.
            unique_pct: Percentage of unique values (0–100).

        Returns:
            True if the column looks like a primary/foreign key or ID.
        """
        name_matches = bool(_ID_NAME_PATTERN.search(col_name))
        highly_unique = unique_pct >= (ID_UNIQUENESS_THRESHOLD * 100)
        return name_matches and highly_unique

    def _sample_values(self, series: pd.Series) -> list[Any]:
        """
        Return up to SAMPLE_SIZE non-null representative values.

        Preference is given to distinct values. If fewer distinct values
        exist than SAMPLE_SIZE, all of them are returned.

        Args:
            series: The raw column series.

        Returns:
            List of Python-native scalar values (not numpy types).
        """
        non_null = series.dropna()
        if non_null.empty:
            return []

        unique_vals = non_null.unique()
        sample = unique_vals[:SAMPLE_SIZE]
        return [self._to_python_scalar(v) for v in sample]

    @staticmethod
    def _try_coerce_datetime(series: pd.Series) -> pd.Series | None:
        """
        Attempt to parse an object/string column as datetime.

        Returns the coerced Series if more than 80% of non-null values
        parse successfully, otherwise returns None.

        Args:
            series: Raw string/object column.

        Returns:
            Coerced datetime Series, or None if coercion rate is too low.
        """
        try:
            coerced = pd.to_datetime(series, errors="coerce")
            non_null_original = series.notna().sum()
            if non_null_original == 0:
                return None
            success_rate = coerced.notna().sum() / non_null_original
            return coerced if success_rate >= 0.80 else None
        except Exception:
            return None

    @staticmethod
    def _safe_scalar(value: Any) -> float | None:
        """
        Convert a numpy scalar to a Python float, returning None for
        NaN or infinite values that would break JSON serialisation.

        Args:
            value: Scalar value from a pandas aggregation.

        Returns:
            Python float, or None.
        """
        try:
            f = float(value)
            return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """
        Convert numpy scalar types to plain Python types so the profile
        dict is fully JSON-serialisable without a custom encoder.

        Args:
            value: A value from series.unique().

        Returns:
            int, float, str, bool, or None.
        """
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value