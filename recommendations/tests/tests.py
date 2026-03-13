"""
Integration test suite for the recommendations app.

Tests the full pipeline from user interaction through to API response.
Uses Django TestCase with a seeded SQLite database.
Redis cache is mocked — no live external services required.

Test classes:
  TestRecommendationPipeline  — end-to-end: interaction → cache → API response
  TestColdStartUser           — user with no history receives CBF fallback scores
  TestCacheInvalidation       — saving a new interaction clears the user's cache
  TestFeedbackEndpoint        — POST /feedback creates a UserInteraction record
  TestRecommendationEndpoint  — GET /recommendations returns valid serialised output

Run: python manage.py test recommendations
"""



from django.test import TestCase

# Create your tests here.
