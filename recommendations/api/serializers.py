"""
DRF serializers for the recommendations API.

Translates between internal domain objects and the JSON representations
returned by the API. One serialiser per major resource:

  RecommendationRequestSerializer
    Validates optional GET params: top_n (int), alpha (float 0–1).

  RecommendedDatasetSerializer
    Shapes a single recommendation: dataset_id, title, rank, s_hybrid.

  RecommendationListSerializer
    Wraps a list of RecommendedDatasetSerializer with metadata:
    alpha, top_n, generated_at.

  FeedbackSerializer
    Validates POST body: dataset_id, interaction_type, rating (optional).
"""

from rest_framework import serializers
from recommendations.models import UserInteraction


class RecommendationRequestSerializer(serializers.Serializer):
    """
    Validates optional GET query parameters for the recommendation endpoint.

    Fields:
        top_n (int): Number of recommendations to return. Defaults to 10.
        alpha (float): Hybrid blending weight between collaborative filtering
                       and content-based filtering. Must be in [0.0, 1.0].
                       Defaults to 0.5.
    """

    top_n = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        help_text="Number of recommendations to return (default: 10).",
    )
    alpha = serializers.FloatField(
        required=False,
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        help_text="Hybrid blending weight between CF and CBF scores (0–1, default: 0.5).",
    )


class RecommendedDatasetSerializer(serializers.Serializer):
    """
    Shapes a single dataset recommendation for API output.

    Exposes the hybrid score (s_hybrid) and rank to consumers.
    Component scores (s_cf, s_cbf) are intentionally excluded from the
    default output; surface them only via a dedicated debug endpoint.

    Fields:
        dataset_id (str): Unique identifier for the dataset.
        title (str): Human-readable dataset title.
        rank (int): Position in the ranked recommendation list (1-based).
        s_hybrid (float): Blended recommendation score.
        confidence (str): Derived human-readable confidence label
                          (computed via SerializerMethodField).
    """

    dataset_id = serializers.CharField(read_only=True)
    title = serializers.CharField(read_only=True)
    rank = serializers.IntegerField(read_only=True, min_value=1)
    s_hybrid = serializers.FloatField(read_only=True)
    confidence = serializers.SerializerMethodField(
        help_text="Human-readable confidence label derived from s_hybrid."
    )

    # Internal component scores — never exposed publicly.
    # Kept as private attributes on the serialiser for potential debug use.
    _s_cf = serializers.FloatField(write_only=True, required=False)
    _s_cbf = serializers.FloatField(write_only=True, required=False)

    def get_confidence(self, obj) -> str:
        """
        Derive a confidence label from the hybrid score.

        Thresholds:
            >= 0.75  → "high"
            >= 0.45  → "medium"
            < 0.45   → "low"
        """
        score = getattr(obj, "s_hybrid", None) or obj.get("s_hybrid", 0)
        if score >= 0.75:
            return "high"
        elif score >= 0.45:
            return "medium"
        return "low"


class RecommendationListSerializer(serializers.Serializer):
    """
    Wraps a ranked list of dataset recommendations with request metadata.

    Fields:
        recommendations (list[RecommendedDatasetSerializer]): Ordered list
            of recommended datasets.
        alpha (float): The alpha value used to blend CF and CBF scores.
        top_n (int): The maximum number of results requested.
        generated_at (datetime): UTC timestamp when the recommendations
            were produced.
    """

    recommendations = RecommendedDatasetSerializer(many=True, read_only=True)
    alpha = serializers.FloatField(read_only=True)
    top_n = serializers.IntegerField(read_only=True)
    generated_at = serializers.DateTimeField(read_only=True)


class FeedbackSerializer(serializers.ModelSerializer):
    """
    Validates the POST body for the feedback endpoint.

    Backed by the UserInteraction model. Enforces allowed interaction types
    and an optional bounded rating.

    Fields:
        dataset_id (str): Required. Identifier of the dataset being rated.
        interaction_type (str): Required. One of the choices defined on
            UserInteraction (e.g. "click", "download", "bookmark").
        rating (int | None): Optional. User satisfaction score in [1, 5].
    """

    rating = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=1,
        max_value=5,
        help_text="Optional satisfaction rating between 1 and 5.",
    )

    class Meta:
        model = UserInteraction
        fields = ["dataset_id", "interaction_type", "rating"]

    def validate_interaction_type(self, value: str) -> str:
        """Ensure interaction_type is one of the model's allowed choices."""
        valid_choices = {
            choice[0] for choice in UserInteraction.INTERACTION_TYPE_CHOICES
        }
        if value not in valid_choices:
            raise serializers.ValidationError(
                f"Invalid interaction type '{value}'. "
                f"Must be one of: {', '.join(sorted(valid_choices))}."
            )
        return value