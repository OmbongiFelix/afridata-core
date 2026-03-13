"""
TF-IDF matrix and item vector storage for the content-based engine.

Handles persistence of scipy sparse matrices produced by the TF-IDF
vectoriser, together with their associated item_id index arrays.

Separate from model_store.py because:
  - TF-IDF matrices are scipy.sparse (not joblib-optimal)
  - They require an item_id index array stored alongside the matrix
  - They can be 100MB+ and need streaming load for memory efficiency

Used by:
  management/commands/train_content_based.py  → save_tfidf_matrix()
  domain/engines/content_based.py             → load_tfidf_matrix()
"""
