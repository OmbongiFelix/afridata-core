"""
URL configuration for the recommendations API.

This is the only urls.py in the app. The project-level urls.py
includes all recommendation routes via:

    path("api/", include("recommendations.api.urls")),

Route map:
  recommendations/          → RecommendationListView  (GET)
  recommendations/feedback/ → FeedbackView            (POST)

app_name = 'recommendations'  (for use with reverse())
"""

