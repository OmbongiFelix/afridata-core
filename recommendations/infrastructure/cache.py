"""
Redis cache helpers for the recommendations app.

Provides a thin wrapper around Django's cache framework for reading
and writing per-user Top-N recommendation lists.

Cache key format:  rec:user:{user_id}
Default TTL:       3600 seconds (1 hour), overridable per call.

Functions:
  get_cached_recommendations(user_id) -> RankedList | None
  set_cached_recommendations(user_id, ranked_list, ttl)
  invalidate_user_cache(user_id)

Configure Redis connection via settings.CACHES['default'].
"""
