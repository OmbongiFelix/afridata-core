"""
ML-based semantic type classifier for dataset columns.

Takes a list of ColumnProfile objects and assigns a semantic_type
label to each column. Semantic types go beyond primitive dtypes to
capture business meaning, e.g.:

    int     → 'age', 'count', 'id', 'year'
    string  → 'email', 'phone', 'address', 'name', 'url', 'category'
    float   → 'currency', 'percentage', 'latitude', 'score'

Uses a lightweight scikit-learn classifier trained on column name
tokens + statistical profile features. Falls back to rule-based
heuristics when model confidence is below threshold.
"""
