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

from django.urls import path
from .views import RecommendationListView, FeedbackView

app_name = 'recommendations'

urlpatterns = [
    path('recommendations/', RecommendationListView.as_view(), name='list'),
    path('recommendations/feedback/', FeedbackView.as_view(), name='feedback'),
]