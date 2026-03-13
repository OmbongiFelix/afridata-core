"""
Django admin registrations for the recommendations app.

Provides a read-friendly admin interface for monitoring user interactions
and recommendation outputs without needing direct database access.

Registered models:
  UserInteraction      — list by user, interaction_type, timestamp
  Dataset              — list by title, category, updated_at
  RecommendationResult — list by user, generated_at; show alpha and item count
"""



from django.contrib import admin

# Register your models here.
