"""
Django ORM query helpers for the recommendations app.

This is the only module in the app that imports from models.py.
Domain code (engines, ranking) must never query the DB directly —
they call functions from here instead.

Functions:
  get_user_interactions(user_id) -> list[UserInteraction]
  get_all_dataset_ids()          -> list[int]
  save_recommendation_result(user_id, ranked_list, alpha) -> RecommendationResult
  get_latest_recommendation(user_id) -> RecommendationResult | None
"""

