from __future__ import annotations

import logging
from rest_framework import status
from rest_framework.generics import CreateAPIView, GenericAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from recommendations.domain.schemas import EngineConfig
from recommendations.domain.engines.hybrid import HybridEngine
from recommendations.infrastructure.cache import (
    get_cached_recommendations,
    set_cached_recommendations,
)
from recommendations.models import DatasetProxy
from .serializers import (
    FeedbackSerializer,
    RecommendationListSerializer,
    RecommendationRequestSerializer,
)

logger = logging.getLogger(__name__)


class RecommendationListView(GenericAPIView):
    """
    GET /api/recommendations/
    GET /api/recommendations/<dataset_id>/

    Returns Top-N personalised or dataset-targeted recommendations for the authenticated user.
    Supports query parameters:
      - strategy: "hybrid" (default), "content", "collaborative"
      - top_n: integer (default: 10)
      - alpha: float in [0.0, 1.0] (default: 0.5)
      - dataset_id: optional target dataset ID
    """

    permission_classes = [IsAuthenticated]
    serializer_class = RecommendationListSerializer

    def get(self, request, *args, **kwargs):
        user = request.user
        user_id = getattr(user, "id", getattr(user, "pk", 1))

        # Validate query parameters
        param_serializer = RecommendationRequestSerializer(data=request.query_params)
        param_serializer.is_valid(raise_exception=False)
        top_n = param_serializer.validated_data.get("top_n", 10)
        alpha = param_serializer.validated_data.get("alpha", 0.5)

        strategy = request.query_params.get("strategy", "hybrid").lower()
        target_dataset_id = kwargs.get("dataset_id") or request.query_params.get("dataset_id")

        cfg = EngineConfig(top_n=top_n, alpha=alpha)

        # Check cache for default hybrid user recommendations
        ranked_list = None
        if strategy == "hybrid" and not target_dataset_id:
            try:
                ranked_list = get_cached_recommendations(user_id=user_id)
            except Exception:
                ranked_list = None

        if ranked_list is None:
            engine = HybridEngine(user=user)
            ranked_list = engine.recommend(
                user_id=user_id,
                strategy=strategy,
                config=cfg,
            )
            if strategy == "hybrid" and not target_dataset_id:
                try:
                    set_cached_recommendations(user_id, ranked_list)
                except Exception:
                    pass

        # Hydrate dataset metadata (titles, categories) from DatasetProxy
        item_ids = [item.item_id for item in ranked_list.items]
        datasets_map = {
            ds.dataset_id: ds
            for ds in DatasetProxy.objects.filter(dataset_id__in=item_ids)
        }

        rec_items = []
        for idx, item in enumerate(ranked_list.items):
            ds = datasets_map.get(item.item_id)
            title = ds.title if ds else f"Dataset {item.item_id}"
            rec_items.append({
                "dataset_id": str(item.item_id),
                "title": title,
                "rank": idx + 1,
                "s_hybrid": round(float(item.s_hybrid), 4),
            })

        payload = {
            "recommendations": rec_items,
            "alpha": float(getattr(ranked_list, "alpha", alpha)),
            "top_n": int(top_n),
            "generated_at": ranked_list.generated_at,
        }

        serializer = self.get_serializer(payload)
        return Response(serializer.data, status=status.HTTP_200_OK)


class FeedbackView(CreateAPIView):
    """
    POST /api/recommendations/feedback/

    Records explicit user feedback (rating or thumbs up/down) as a
    UserInteraction. Saving the interaction triggers cache invalidation
    via Django signals.
    """

    permission_classes = [IsAuthenticated]
    serializer_class = FeedbackSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)