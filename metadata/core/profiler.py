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
 