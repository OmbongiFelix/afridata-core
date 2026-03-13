"""
This module contains the application configuration for the recommendations app.
"""
from django.apps import AppConfig

"""add a ready() method that imports and connects your signals.py. Without it, signals never fire 
regardless of what's in the file"""

class RecommendationsConfig(AppConfig):
    name = 'recommendations'
