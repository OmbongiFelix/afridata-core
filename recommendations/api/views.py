"""
API views for the recommendations app.

Provides RESTful endpoints to retrieve personalised recommendations
and submit explicit user feedback. Uses DRF GenericAPIView.

Endpoints (registered in api/urls.py):
  GET  /api/recommendations/
    Returns Top-N recommended datasets for the authenticated user.
    Reads from cache first; falls back to a live HybridEngine call.

  POST /api/recommendations/feedback/
    Records explicit user feedback (rating, thumbs up/down) as a
    UserInteraction, which triggers cache invalidation via signals.

Views contain no scoring or ranking logic.
All recommendation computation is delegated to the domain layer.
"""

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from recommendations.infrastructure.cache import get_cached_recommendations
from recommendations.domain.engines.hybrid import HybridEngine
from .serializers import RecommendationListSerializer, FeedbackSerializer


class RecommendationListView(ListAPIView):
    """
    GET /api/recommendations/

    Returns Top-N personalised recommended datasets for the authenticated user.

    Strategy:
      1. Check the cache for a pre-computed ranked list.
      2. On a cache miss, delegate to HybridEngine for a live computation.
         For large/expensive requests this should be enqueued as a Celery task;
         the synchronous fallback here is intentionally lightweight.
    """

    permission_classes = [IsAuthenticated]
    serializer_class = RecommendationListSerializer

    def list(self, request, *args, **kwargs):
        user = request.user

        ranked_list = get_cached_recommendations(user_id=user.pk)

        if ranked_list is None:
            engine = HybridEngine(user=user)
            ranked_list = engine.get_recommendations()

        serializer = self.get_serializer(ranked_list)
        return Response(serializer.data, status=status.HTTP_200_OK)


class FeedbackView(CreateAPIView):
    """
    POST /api/recommendations/feedback/

    Records explicit user feedback (rating or thumbs up/down) as a
    UserInteraction. Saving the interaction triggers cache invalidation
    via Django signals — no manual invalidation is needed here.

    Returns 201 on success with the serialised interaction payload.
    """

    permission_classes = [IsAuthenticated]
    serializer_class = FeedbackSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)