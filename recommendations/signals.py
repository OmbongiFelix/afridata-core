"""
Django signal receivers for the recommendations app.

Receivers:

  on_interaction_saved  — fires on post_save of UserInteraction.
                          Enqueues tasks.refresh_user_scores(user_id)
                          to invalidate and recompute the user's cache.

  on_interaction_deleted — fires on post_delete of UserInteraction.
                           Same invalidation path as above.

Signals are connected inside AppConfig.ready() in apps.py.
Do not import this module directly anywhere else.
"""
