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
from rest_framework.test import APIClient
from unittest.mock import patch, MagicMock
from recommendations.models import UserInteraction, Dataset

import factory
from django.contrib.auth import get_user_model

User = get_user_model()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user_{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    password = factory.PostGenerationMethodCall("set_password", "testpass123")


class DatasetFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Dataset

    title = factory.Sequence(lambda n: f"Item {n}")
    description = factory.Faker("sentence")
    # Add / adjust fields to match your actual Dataset model definition


class UserInteractionFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = UserInteraction

    user = factory.SubFactory(UserFactory)
    item = factory.SubFactory(DatasetFactory)
    rating = factory.Faker("pyfloat", min_value=1.0, max_value=5.0, right_digits=1)
    # Add / adjust fields to match your actual UserInteraction model definition


# ---------------------------------------------------------------------------
# TestRecommendationPipeline
# ---------------------------------------------------------------------------

class TestRecommendationPipeline(TestCase):
    """End-to-end: user with history → cache → API response returns ranked list."""

    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)

        # Seed interactions so the user is not a cold-start user
        self.items = DatasetFactory.create_batch(5)
        for item in self.items:
            UserInteractionFactory(user=self.user, item=item)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_user_with_history_receives_nonempty_ranked_list(
        self, mock_cf, mock_set_cache, mock_get_cache
    ):
        """A user with interaction history must receive a non-empty ranked list."""
        mock_cf.return_value = [
            {"item_id": item.pk, "score": 0.9 - (i * 0.1)}
            for i, item in enumerate(self.items)
        ]

        response = self.client.get("/recommendations/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_ranked_list_items_have_required_fields(
        self, mock_cf, mock_set_cache, mock_get_cache
    ):
        """Each item in the ranked list must expose required serialised fields."""
        mock_cf.return_value = [
            {"item_id": self.items[0].pk, "score": 0.95}
        ]

        response = self.client.get("/recommendations/")
        self.assertEqual(response.status_code, 200)

        first = response.json()["recommendations"][0]
        for field in ("item_id", "score"):
            self.assertIn(field, first, msg=f"Missing field: {field}")

    @patch("recommendations.infrastructure.cache.get_user_cache")
    @patch("recommendations.infrastructure.cache.set_user_cache")
    def test_cached_response_is_returned_without_calling_engine(
        self, mock_set_cache, mock_get_cache
    ):
        """When a warm cache entry exists the engine must not be invoked."""
        cached_payload = {
            "recommendations": [{"item_id": self.items[0].pk, "score": 0.88}]
        }
        mock_get_cache.return_value = cached_payload

        with patch("recommendations.domain.engine.run_collaborative_filter") as mock_cf:
            response = self.client.get("/recommendations/")
            mock_cf.assert_not_called()

        self.assertEqual(response.status_code, 200)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_results_are_ordered_by_descending_score(
        self, mock_cf, mock_set_cache, mock_get_cache
    ):
        """Recommendations must be returned in descending score order."""
        mock_cf.return_value = [
            {"item_id": self.items[i].pk, "score": round(0.9 - i * 0.1, 1)}
            for i in range(5)
        ]

        response = self.client.get("/recommendations/")
        self.assertEqual(response.status_code, 200)

        scores = [r["score"] for r in response.json()["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# TestColdStartUser
# ---------------------------------------------------------------------------

class TestColdStartUser(TestCase):
    """A user with no interaction history must receive CBF fallback scores."""

    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)

        # Populate catalogue so CBF has items to score
        self.items = DatasetFactory.create_batch(3)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_content_based_filter")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_cold_start_user_receives_cbf_recommendations(
        self, mock_cf, mock_cbf, mock_set_cache, mock_get_cache
    ):
        """Cold-start user must receive content-based fallback, not an error."""
        mock_cbf.return_value = [
            {"item_id": item.pk, "score": 0.5} for item in self.items
        ]

        response = self.client.get("/recommendations/")

        mock_cf.assert_not_called()
        mock_cbf.assert_called_once()
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_content_based_filter")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_cold_start_response_is_not_an_error(
        self, mock_cf, mock_cbf, mock_set_cache, mock_get_cache
    ):
        """Cold-start path must not return a 4xx or 5xx status code."""
        mock_cbf.return_value = [{"item_id": self.items[0].pk, "score": 0.5}]

        response = self.client.get("/recommendations/")
        self.assertLess(response.status_code, 400)


# ---------------------------------------------------------------------------
# TestCacheInvalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation(TestCase):
    """Saving a new UserInteraction must clear the user's recommendation cache."""

    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)
        self.item = DatasetFactory()

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_new_interaction_invalidates_cache(self, mock_invalidate):
        """POST /feedback must call cache.invalidate_user_cache() for the user."""
        payload = {"item_id": self.item.pk, "rating": 4.0}
        response = self.client.post("/feedback/", data=payload, format="json")

        self.assertIn(response.status_code, (200, 201))
        mock_invalidate.assert_called_once_with(self.user.pk)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_invalidation_not_called_on_bad_request(self, mock_invalidate):
        """Cache must not be invalidated when the feedback payload is invalid."""
        response = self.client.post("/feedback/", data={}, format="json")

        self.assertEqual(response.status_code, 400)
        mock_invalidate.assert_not_called()


# ---------------------------------------------------------------------------
# TestRecommendationEndpoint
# ---------------------------------------------------------------------------

class TestRecommendationEndpoint(TestCase):
    """GET /recommendations/ must return HTTP 200 with valid serialised output."""

    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)
        self.items = DatasetFactory.create_batch(3)
        for item in self.items:
            UserInteractionFactory(user=self.user, item=item)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_endpoint_returns_http_200(self, mock_cf, mock_set, mock_get):
        mock_cf.return_value = [{"item_id": self.items[0].pk, "score": 0.8}]

        response = self.client.get("/recommendations/")
        self.assertEqual(response.status_code, 200)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_response_matches_recommendation_list_serializer_shape(
        self, mock_cf, mock_set, mock_get
    ):
        """Response JSON must conform to RecommendationListSerializer shape."""
        mock_cf.return_value = [{"item_id": self.items[0].pk, "score": 0.8}]

        response = self.client.get("/recommendations/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIn("recommendations", data)
        self.assertIsInstance(data["recommendations"], list)

    @patch("recommendations.infrastructure.cache.get_user_cache", return_value=None)
    @patch("recommendations.infrastructure.cache.set_user_cache")
    @patch("recommendations.domain.engine.run_collaborative_filter")
    def test_each_recommendation_item_has_item_id_and_score(
        self, mock_cf, mock_set, mock_get
    ):
        """Every recommendation entry must contain item_id and score fields."""
        mock_cf.return_value = [
            {"item_id": item.pk, "score": 0.7} for item in self.items
        ]

        response = self.client.get("/recommendations/")
        self.assertEqual(response.status_code, 200)

        for rec in response.json()["recommendations"]:
            self.assertIn("item_id", rec)
            self.assertIn("score", rec)

    def test_unauthenticated_request_is_rejected(self):
        """Unauthenticated clients must not receive recommendations."""
        unauthenticated = APIClient()
        response = unauthenticated.get("/recommendations/")
        self.assertIn(response.status_code, (401, 403))


# ---------------------------------------------------------------------------
# TestFeedbackEndpoint
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint(TestCase):
    """POST /feedback/ must return HTTP 201 and create a UserInteraction record."""

    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)
        self.item = DatasetFactory()

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_valid_feedback_returns_201(self, mock_invalidate):
        payload = {"item_id": self.item.pk, "rating": 3.5}
        response = self.client.post("/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, 201)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_valid_feedback_creates_user_interaction_record(self, mock_invalidate):
        """A successful POST must persist exactly one new UserInteraction row."""
        before = UserInteraction.objects.filter(user=self.user).count()

        payload = {"item_id": self.item.pk, "rating": 4.0}
        self.client.post("/feedback/", data=payload, format="json")

        after = UserInteraction.objects.filter(user=self.user).count()
        self.assertEqual(after, before + 1)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_created_interaction_belongs_to_authenticated_user(self, mock_invalidate):
        """The created record must be associated with the requesting user."""
        payload = {"item_id": self.item.pk, "rating": 5.0}
        self.client.post("/feedback/", data=payload, format="json")

        interaction = UserInteraction.objects.filter(
            user=self.user, item=self.item
        ).last()
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction.user, self.user)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_missing_item_id_returns_400(self, mock_invalidate):
        """Payload without item_id must be rejected with HTTP 400."""
        payload = {"rating": 3.0}
        response = self.client.post("/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, 400)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_invalid_rating_value_returns_400(self, mock_invalidate):
        """Out-of-range or non-numeric rating must be rejected with HTTP 400."""
        payload = {"item_id": self.item.pk, "rating": "not-a-number"}
        response = self.client.post("/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, 400)

    @patch("recommendations.infrastructure.cache.invalidate_user_cache")
    def test_nonexistent_item_returns_404(self, mock_invalidate):
        """Feedback referencing a non-existent item must return HTTP 404."""
        payload = {"item_id": 999999, "rating": 3.0}
        response = self.client.post("/feedback/", data=payload, format="json")
        self.assertEqual(response.status_code, 404)

    def test_unauthenticated_feedback_is_rejected(self):
        """Unauthenticated clients must not be able to post feedback."""
        unauthenticated = APIClient()
        payload = {"item_id": self.item.pk, "rating": 3.0}
        response = unauthenticated.post("/feedback/", data=payload, format="json")
        self.assertIn(response.status_code, (401, 403))