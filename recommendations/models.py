"""
Database models for the recommendations app.

Three models cover the full lifecycle of a recommendation:

  UserInteraction  — one record per user action on a dataset
                     (view, download, bookmark, explicit rating).
                     Primary training signal for collaborative filtering.

  Dataset          — lightweight metadata proxy kept in sync with the
                     datasets app. Provides title, description, tags,
                     and category for TF-IDF content-based scoring.

  RecommendationResult — persisted Top-N recommendation list per user.
                         Stores ranked item IDs, fused scores, the alpha
                         weight used, and generation timestamp.

Models are data containers only. No scoring or ranking logic here.
"""


from django.db import models

# Create your models here.
