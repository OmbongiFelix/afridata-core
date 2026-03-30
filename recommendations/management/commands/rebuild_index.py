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

import time
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from recommendations.infrastructure.cache import invalidate_user_cache
from recommendations.infrastructure import persistence
from recommendations import tasks


class Command(BaseCommand):
    help = (
        "Invalidates all user recommendation caches and recomputes Top-N scores "
        "using the latest trained models."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--users",
            type=str,
            default=None,
            help=(
                "Comma-separated list of user IDs to rebuild. "
                "Defaults to all active users (at least one interaction in last 90 days)."
            ),
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help=(
                "Fusion weight to use for this rebuild. "
                "Defaults to settings.RECOMMENDATIONS_ALPHA."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Log affected user IDs without writing to cache or enqueueing tasks.",
        )

    def handle(self, *args, **options) -> None:
        start_time = time.monotonic()

        dry_run: bool = options["dry_run"]
        alpha: float = options["alpha"] if options["alpha"] is not None else getattr(
            settings, "RECOMMENDATIONS_ALPHA", 0.5
        )

        # Resolve target user IDs
        if options["users"]:
            try:
                user_ids = [int(uid.strip()) for uid in options["users"].split(",") if uid.strip()]
            except ValueError as exc:
                self.stderr.write(self.style.ERROR(f"Invalid --users value: {exc}"))
                return
            self.stdout.write(
                f"Rebuilding index for {len(user_ids)} explicitly specified user(s)."
            )
        else:
            cutoff = timezone.now() - timedelta(days=90)
            user_ids = persistence.get_active_user_ids(since=cutoff)
            self.stdout.write(
                f"Found {len(user_ids)} active user(s) with interactions in the last 90 days."
            )

        if not user_ids:
            self.stdout.write(self.style.WARNING("No users to rebuild. Exiting."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no cache writes or tasks enqueued."))
            for uid in user_ids:
                self.stdout.write(f"  [dry-run] Would rebuild user_id={uid}")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Dry run complete. {len(user_ids)} user(s) would have been rebuilt."
                )
            )
            return

        total = len(user_ids)
        failed = []

        for idx, user_id in enumerate(user_ids, start=1):
            try:
                invalidate_user_cache(user_id)
                tasks.refresh_user_scores.delay(user_id, alpha=alpha)
                self.stdout.write(f"  [{idx}/{total}] Enqueued rebuild for user_id={user_id}")
            except Exception as exc:  # noqa: BLE001
                failed.append(user_id)
                self.stderr.write(
                    self.style.ERROR(
                        f"  [{idx}/{total}] Failed for user_id={user_id}: {exc}"
                    )
                )

        elapsed = time.monotonic() - start_time
        succeeded = total - len(failed)

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. {succeeded}/{total} users enqueued successfully in {elapsed:.2f}s "
                f"(alpha={alpha})."
            )
        )

        if failed:
            self.stderr.write(
                self.style.ERROR(
                    f"{len(failed)} user(s) failed: {failed}"
                )
            )