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

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from recommendations.models import UserInteraction
from recommendations import tasks


@receiver(
    post_save,
    sender=UserInteraction,
    dispatch_uid="recommendations.signals.on_interaction_saved",
)
def on_interaction_saved(sender, instance, created, **kwargs):
    """
    Fires on post_save of UserInteraction.

    Enqueues tasks.refresh_user_scores to invalidate and recompute the
    user's recommendation cache whenever an interaction is created or
    updated. Heavy work is always delegated to Celery — this handler
    must not block.

    Args:
        sender:   The UserInteraction model class.
        instance: The UserInteraction instance that was saved.
        created:  True if a new record was inserted; False on update.
        **kwargs: Additional signal keyword arguments (ignored).
    """
    tasks.refresh_user_scores.delay(instance.user_id)


@receiver(
    post_delete,
    sender=UserInteraction,
    dispatch_uid="recommendations.signals.on_interaction_deleted",
)
def on_interaction_deleted(sender, instance, **kwargs):
    """
    Fires on post_delete of UserInteraction.

    Enqueues tasks.refresh_user_scores to invalidate and recompute the
    user's recommendation cache whenever an interaction is removed.
    Same invalidation path as on_interaction_saved. Heavy work is always
    delegated to Celery — this handler must not block.

    Args:
        sender:   The UserInteraction model class.
        instance: The UserInteraction instance that was deleted.
        **kwargs: Additional signal keyword arguments (ignored).
    """
    tasks.refresh_user_scores.delay(instance.user_id)