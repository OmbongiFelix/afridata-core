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
"""
