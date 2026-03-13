"""
DRF serializers for the recommendations API.

Translates between internal domain objects and the JSON representations
returned by the API. One serialiser per major resource:

  RecommendationRequestSerializer
    Validates optional GET params: top_n (int), alpha (float 0–1).

  RecommendedDatasetSerializer
    Shapes a single recommendation: dataset_id, title, rank, s_hybrid.

  RecommendationListSerializer
    Wraps a list of RecommendedDatasetSerializer with metadata:
    alpha, top_n, generated_at.

  FeedbackSerializer
    Validates POST body: dataset_id, interaction_type, rating (optional).
"""
