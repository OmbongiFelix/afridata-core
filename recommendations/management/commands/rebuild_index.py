"""
Management command: python manage.py rebuild_index

Invalidates all existing recommendation caches and triggers a full
recompute of Top-N scores for every active user using the latest
trained models.

Run after either training command completes to ensure cached results
reflect the new model weights.

Options:
  --users   Comma-separated user IDs to rebuild (default: all active users)
  --alpha   Fusion weight for this rebuild (default: settings.RECOMMENDATIONS_ALPHA)
  --dry-run Log what would be rebuilt without writing to cache
"""
