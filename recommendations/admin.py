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
from .models import UserInteraction, Dataset, RecommendationResult


@admin.register(UserInteraction)
class UserInteractionAdmin(admin.ModelAdmin):
    list_display = ("user", "interaction_type", "timestamp")
    list_filter = ("interaction_type",)
    search_fields = ("user__username", "user__email")
    readonly_fields = ("user", "interaction_type", "timestamp")


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "updated_at")
    list_filter = ("category",)
    readonly_fields = ("title", "category", "updated_at")


@admin.register(RecommendationResult)
class RecommendationResultAdmin(admin.ModelAdmin):
    list_display = ("user", "generated_at", "alpha", "item_count")
    readonly_fields = ("user", "generated_at", "alpha", "ranked_items_preview")

    def item_count(self, obj):
        """Return the number of items in ranked_items."""
        if isinstance(obj.ranked_items, list):
            return len(obj.ranked_items)
        return 0
    item_count.short_description = "Item Count"

    def ranked_items_preview(self, obj):
        """Read-only JSON preview of ranked_items."""
        import json
        try:
            return json.dumps(obj.ranked_items, indent=2)
        except (TypeError, ValueError):
            return str(obj.ranked_items)
    ranked_items_preview.short_description = "Ranked Items (JSON Preview)"


    