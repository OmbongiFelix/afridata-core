"""
Collaborative Filtering engine using Matrix Factorisation.

Loads pre-trained model weights from infrastructure/model_store.py
and scores a list of candidate items for a given user.

Algorithm: Alternating Least Squares (ALS) or truncated SVD.
           Configured via settings.CF_MODEL_TYPE.

Cold-start handling:
  Users with no interaction history receive uniform zero S_CF scores.
  The hybrid engine then falls back entirely to S_CBF for cold users.

Does not train. Training is handled by:
  management/commands/train_collaborative.py
"""
