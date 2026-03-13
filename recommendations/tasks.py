"""
Celery task definitions for the recommendations app.

Tasks:

  refresh_user_scores(user_id)
    Recomputes and caches Top-N recommendations for one user.
    Called by signals when a UserInteraction is created or deleted.

  train_collaborative_task()
    Full refit of the collaborative filter from interaction history.
    Triggered nightly via Celery beat or by the management command.

  train_content_based_task()
    Rebuilds the TF-IDF matrix from current Dataset metadata.
    Run after bulk dataset metadata updates.

All tasks must be idempotent and safe to retry on failure.
"""
