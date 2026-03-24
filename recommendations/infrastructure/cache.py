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

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from django.core.cache import cache

from recommendations.domain.schemas import RankedList, ScoredCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_KEY_PREFIX = "rec:user:"
_DEFAULT_TTL = 3600  # seconds


def _cache_key(user_id: int) -> str:
    """Return the namespaced cache key for a given user."""
    return f"{_KEY_PREFIX}{user_id}"


def _serialise(ranked_list: RankedList) -> str:
    """
    Serialise a RankedList to a JSON string.

    Uses plain JSON (not pickle) so cached data is safe to read across
    Python versions and deployments.
    """
    payload: dict[str, Any] = {
        "user_id": ranked_list.user_id,
        "generated_at": ranked_list.generated_at.isoformat(),
        "items": [
            {
                "item_id": item.item_id,
                "s_cf": item.s_cf,
                "s_cbf": item.s_cbf,
                "s_hybrid": item.s_hybrid,
            }
            for item in ranked_list.items
        ],
    }
    return json.dumps(payload)


def _deserialise(raw: str) -> RankedList:
    """
    Deserialise a JSON string back into a RankedList dataclass.

    Raises
    ------
    ValueError
        If the JSON is malformed or missing required fields.
    """
    data = json.loads(raw)
    items = [
        ScoredCandidate(
            item_id=entry["item_id"],
            s_cf=entry["s_cf"],
            s_cbf=entry["s_cbf"],
            s_hybrid=entry["s_hybrid"],
        )
        for entry in data["items"]
    ]
    generated_at = datetime.fromisoformat(data["generated_at"])
    # Ensure timezone-aware if it was stored as UTC naive
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)

    return RankedList(
        user_id=data["user_id"],
        items=items,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cached_recommendations(user_id: int) -> RankedList | None:
    """
    Retrieve the cached Top-N recommendation list for a user.

    Returns ``None`` on a cache miss or if the cache backend is
    unavailable — callers should fall back to a live engine run in
    either case.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.

    Returns
    -------
    RankedList | None
        The deserialised ranked list, or ``None`` on miss / error.
    """
    key = _cache_key(user_id)
    try:
        raw = cache.get(key)
    except Exception:
        logger.warning(
            "Cache backend unavailable during get — user_id=%s", user_id,
            exc_info=True,
        )
        return None

    if raw is None:
        logger.debug("Cache miss — user_id=%s", user_id)
        return None

    try:
        ranked_list = _deserialise(raw)
    except (KeyError, ValueError, json.JSONDecodeError):
        logger.warning(
            "Corrupt cache entry for user_id=%s; treating as miss", user_id,
            exc_info=True,
        )
        return None

    logger.debug("Cache hit — user_id=%s", user_id)
    return ranked_list


def set_cached_recommendations(
    user_id: int,
    ranked_list: RankedList,
    ttl: int = _DEFAULT_TTL,
) -> None:
    """
    Write a Top-N recommendation list to the cache.

    Silently logs a warning and continues if the cache backend is
    unavailable — a failed write must never crash the request path.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.
    ranked_list:
        The :class:`~recommendations.domain.schemas.RankedList` to cache.
    ttl:
        Time-to-live in seconds.  Defaults to 3600 (1 hour).
    """
    key = _cache_key(user_id)
    try:
        cache.set(key, _serialise(ranked_list), timeout=ttl)
        logger.debug(
            "Cache written — user_id=%s ttl=%ss items=%d",
            user_id,
            ttl,
            len(ranked_list.items),
        )
    except Exception:
        logger.warning(
            "Cache backend unavailable during set — user_id=%s", user_id,
            exc_info=True,
        )


def invalidate_user_cache(user_id: int) -> None:
    """
    Delete the cached recommendation list for a user.

    Called by signal receivers whenever the user logs a new interaction,
    so the next request triggers a fresh engine run rather than serving
    a stale result.

    Silently logs a warning and continues if the cache backend is
    unavailable.

    Parameters
    ----------
    user_id:
        Primary key of the user in AUTH_USER_MODEL.
    """
    key = _cache_key(user_id)
    try:
        cache.delete(key)
        logger.debug("Cache invalidated — user_id=%s", user_id)
    except Exception:
        logger.warning(
            "Cache backend unavailable during delete — user_id=%s", user_id,
            exc_info=True,
        )