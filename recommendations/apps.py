"""
Django AppConfig for the recommendations app.

The ready() method is the single authoritative place that connects
all signal receivers. Without it, signals.py is never imported and
score invalidation silently stops working.

Usage (auto-loaded via default_app_config or INSTALLED_APPS):

    INSTALLED_APPS = [
        ...
        "recommendations.apps.RecommendationsConfig",
    ]
"""



from django.apps import AppConfig

"""add a ready() method that imports and connects your signals.py. Without it, signals never fire 
regardless of what's in the file"""

class RecommendationsConfig(AppConfig):
    name = 'recommendations'
