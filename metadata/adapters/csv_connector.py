"""
Adapter for ingesting CSV data sources.

Reads one or more CSV files, handles encoding detection, delimiter
sniffing, and malformed-row recovery. Returns a clean pd.DataFrame
tagged with source metadata (filename, row_count, ingested_at).

Supports:
    - Local file paths
    - Glob patterns  (e.g. './data/*.csv')
    - Remote HTTP URLs (delegated to requests)

"""