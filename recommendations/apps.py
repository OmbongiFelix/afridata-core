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


class RecommendationsConfig(AppConfig):
    name = "recommendations"

    def ready(self) -> None:
        import recommendations.signals  # noqa: F401