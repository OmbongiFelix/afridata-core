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
