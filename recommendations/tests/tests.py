"""
Integration test suite for the recommendations app.

Tests the full pipeline from user interaction through to API response.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient
from unittest.mock import patch

from recommendations.models import DatasetProxy, UserInteraction, InteractionType
from recommendations.domain.schemas import RankedList, ScoredCandidate

User = get_user_model()


class TestRecommendationPipeline(TestCase):
    """End-to-end: user with history → cache → API response returns ranked list."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client.force_authenticate(user=self.user)

        # Seed datasets
        self.datasets = []
        for i in range(1, 6):
            ds = DatasetProxy.objects.create(
                dataset_id=100 + i,
                title=f"Sample Dataset {i}",
                description=f"Description for dataset {i}",
                tags="health,analytics,africa",
                category="health",
                formats="csv",
                is_active=True,
            )
            self.datasets.append(ds)

        # Seed interactions
        UserInteraction.objects.create(
            user=self.user,
            dataset_id=101,
            interaction_type=InteractionType.VIEW,
            explicit_rating=5.0,
        )

    def test_user_receives_recommendations(self):
        response = self.client.get("/api/recommendations/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertIn("top_n", data)
        self.assertIn("alpha", data)
        self.assertGreater(len(data["recommendations"]), 0)

    def test_strategy_parameter_selection(self):
        for strat in ("content", "collaborative", "hybrid"):
            with self.subTest(strategy=strat):
                response = self.client.get(f"/api/recommendations/?strategy={strat}")
                self.assertEqual(response.status_code, status.HTTP_200_OK)
                data = response.json()
                self.assertIn("recommendations", data)

    def test_recommendation_item_fields(self):
        response = self.client.get("/api/recommendations/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        first = response.json()["recommendations"][0]
        for field in ("dataset_id", "title", "rank", "s_hybrid", "confidence"):
            self.assertIn(field, first, msg=f"Missing field: {field}")

    def test_unauthenticated_request_is_rejected(self):
        client = APIClient()
        response = client.get("/api/recommendations/")
        self.assertIn(response.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))


class TestFeedbackEndpoint(TestCase):
    """POST /api/recommendations/feedback/ creates a UserInteraction record."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="feedbackuser", password="password123")
        self.client.force_authenticate(user=self.user)
        self.dataset = DatasetProxy.objects.create(
            dataset_id=201,
            title="Feedback Target Dataset",
            description="Testing feedback recording",
            is_active=True,
        )

    def test_valid_feedback_creates_interaction(self):
        payload = {
            "dataset_id": "201",
            "interaction_type": "download",
            "rating": 5,
        }
        response = self.client.post("/api/recommendations/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(
            UserInteraction.objects.filter(user=self.user, dataset_id=201, interaction_type="download").exists()
        )

    def test_invalid_interaction_type_rejected(self):
        payload = {
            "dataset_id": "201",
            "interaction_type": "invalid_type_xyz",
            "rating": 5,
        }
        response = self.client.post("/api/recommendations/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_unauthenticated_feedback_rejected(self):
        client = APIClient()
        payload = {
            "dataset_id": "201",
            "interaction_type": "view",
        }
        response = client.post("/api/recommendations/feedback/", data=payload, format="json")
        self.assertIn(response.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))