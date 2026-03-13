"""
Offline evaluation metrics for the recommendations domain.

Used by management commands and CI pipelines to measure model quality
against a held-out test set after every retraining run.
Not called in the live request path.

Functions:
  precision_at_k(recommended, relevant, k) -> float
  recall_at_k(recommended, relevant, k)    -> float
  ndcg_at_k(recommended, relevant, k)      -> float
  evaluate_engine(engine, test_interactions, k=10) -> dict
    Runs all three metrics and returns a summary dict.
"""
