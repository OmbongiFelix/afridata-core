"""
Content-Based Filtering engine using TF-IDF and Cosine Similarity.

Loads a precomputed TF-IDF item matrix from infrastructure/vector_store.py
and scores candidates by cosine similarity to a user profile vector.

User profile construction:
  The profile vector is the weighted average of TF-IDF vectors for all
  items the user has interacted with. Weights are determined by
  UserInteraction.weight (download > view > implicit).

Cold-start handling:
  Users with no interactions receive scores based on global item popularity
  rather than a personal profile vector.

Does not rebuild the matrix. Rebuilding is handled by:
  management/commands/train_content_based.py
"""
