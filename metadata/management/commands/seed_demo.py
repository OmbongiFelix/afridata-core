"""
Management command: seed_demo

Sets up a complete test environment for end-to-end REST verification:
  1. Creates a 'pipeline_admin' group if it doesn't exist
  2. Creates an admin user (default: admin / afridata2024) in that group
  3. Creates a DRF API Token for the admin user
  4. Seeds DatasetProxy records for testing recommendations
  5. Prints the token and all endpoint URLs

Usage:
    uv run python manage.py seed_demo
    uv run python manage.py seed_demo --username alice --password secret123
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand

User = get_user_model()


class Command(BaseCommand):
    help = "Seed demo user, token, and dataset proxies for end-to-end API testing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--username",
            default="admin",
            help="Username for the demo admin user (default: admin)",
        )
        parser.add_argument(
            "--password",
            default="afridata2024",
            help="Password for the demo admin user (default: afridata2024)",
        )
        parser.add_argument(
            "--reset-token",
            action="store_true",
            default=False,
            help="Delete and recreate the API token (generates a new token value).",
        )

    def handle(self, *args, **options):
        from rest_framework.authtoken.models import Token

        username = options["username"]
        password = options["password"]
        reset_token = options["reset_token"]

        # --- 1. pipeline_admin group ---
        group, created = Group.objects.get_or_create(name="pipeline_admin")
        if created:
            self.stdout.write(self.style.SUCCESS(f"  Created group: pipeline_admin"))
        else:
            self.stdout.write(f"  Group 'pipeline_admin' already exists.")

        # --- 2. Admin user ---
        user, user_created = User.objects.get_or_create(username=username)
        if user_created:
            user.set_password(password)
            user.is_staff = True
            user.save()
            self.stdout.write(self.style.SUCCESS(f"  Created user: {username}"))
        else:
            self.stdout.write(f"  User '{username}' already exists.")

        # Ensure the user is in the pipeline_admin group
        if not user.groups.filter(name="pipeline_admin").exists():
            user.groups.add(group)
            self.stdout.write(self.style.SUCCESS(f"  Added {username} to pipeline_admin group."))

        # --- 3. API Token ---
        if reset_token:
            Token.objects.filter(user=user).delete()
            self.stdout.write(f"  Deleted existing token for {username}.")

        token, token_created = Token.objects.get_or_create(user=user)
        if token_created:
            self.stdout.write(self.style.SUCCESS(f"  Created API token."))
        else:
            self.stdout.write(f"  Token for '{username}' already exists.")

        # --- 4. Seed DatasetProxy records ---
        try:
            from recommendations.models import DatasetProxy

            sample_datasets = [
                {
                    "dataset_id": 1001,
                    "title": "Kenya Health Indicators 2023",
                    "description": "Health statistics for Kenya including malaria, HIV, and maternal health.",
                    "tags": "health,kenya,malaria,hiv,maternal",
                    "category": "health",
                    "formats": "csv",
                    "is_active": True,
                },
                {
                    "dataset_id": 1002,
                    "title": "Nigeria Agricultural Production",
                    "description": "Crop yield data for Nigeria across major agricultural zones.",
                    "tags": "agriculture,nigeria,crops,food",
                    "category": "agriculture",
                    "formats": "excel",
                    "is_active": True,
                },
                {
                    "dataset_id": 1003,
                    "title": "Ghana Economic Indicators",
                    "description": "GDP, inflation, and trade data for Ghana 2020-2023.",
                    "tags": "economics,ghana,gdp,trade",
                    "category": "economics",
                    "formats": "csv",
                    "is_active": True,
                },
                {
                    "dataset_id": 1004,
                    "title": "South Africa Census 2022",
                    "description": "Population demographics and household survey data.",
                    "tags": "demographics,south-africa,census,population",
                    "category": "demographics",
                    "formats": "csv",
                    "is_active": True,
                },
                {
                    "dataset_id": 1005,
                    "title": "Ethiopia Climate Data",
                    "description": "Temperature, rainfall, and drought index for Ethiopian regions.",
                    "tags": "climate,ethiopia,temperature,rainfall",
                    "category": "climate",
                    "formats": "csv",
                    "is_active": True,
                },
            ]

            created_count = 0
            for ds_data in sample_datasets:
                _, created = DatasetProxy.objects.get_or_create(
                    dataset_id=ds_data["dataset_id"],
                    defaults={k: v for k, v in ds_data.items() if k != "dataset_id"},
                )
                if created:
                    created_count += 1

            if created_count:
                self.stdout.write(
                    self.style.SUCCESS(f"  Seeded {created_count} DatasetProxy records.")
                )
            else:
                self.stdout.write(f"  DatasetProxy records already exist.")

        except Exception as exc:
            self.stdout.write(
                self.style.WARNING(f"  Could not seed DatasetProxy records: {exc}")
            )

        # --- 5. Print summary ---
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("  DEMO ENVIRONMENT READY"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(f"  Username : {username}")
        self.stdout.write(f"  Password : {password}")
        self.stdout.write(f"  API Token: {token.key}")
        self.stdout.write("")
        self.stdout.write("  Quick-start commands:")
        self.stdout.write("")
        self.stdout.write("  # Set token in shell:")
        self.stdout.write(f'  $TOKEN = "{token.key}"')
        self.stdout.write("")
        self.stdout.write("  # 1. Ingest a CSV dataset:")
        self.stdout.write('  # (PowerShell)')
        self.stdout.write(
            '  Invoke-RestMethod -Uri "http://localhost:8000/api/metadata/runs/"'
            ' -Method POST'
            ' -Headers @{Authorization="Token $TOKEN"; "Content-Type"="application/json"}'
            r' -Body \'{"source": "csv", "source_path": "metadata/tests/fixtures/sample.csv",'
            r' "dataset_title": "AfriData Sample"}\''
        )
        self.stdout.write("")
        self.stdout.write("  # 2. Get recommendations:")
        self.stdout.write(
            '  Invoke-RestMethod -Uri "http://localhost:8000/api/recommendations/?strategy=hybrid"'
            ' -Headers @{Authorization="Token $TOKEN"}'
        )
        self.stdout.write(self.style.SUCCESS("=" * 60))
