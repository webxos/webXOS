import logging
import redis
import os
import json
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages response caching for Vial MCP using Redis."""
    def __init__(self):
        """Initialize CacheManager with Redis connection."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        logger.info("CacheManager initialized")

    def cache_response(self, key: str, response: dict, ttl: int = 300) -> None:
        """Cache a response in Redis with a specified TTL.

        Args:
            key (str): Cache key (e.g., 'notes:wallet_123:postgres').
            response (dict): Response data to cache.
            ttl (int): Time-to-live in seconds (default: 300).

        Raises:
            HTTPException: If caching fails.
        """
        try:
            self.redis_client.setex(key, ttl, json.dumps(response))
            logger.info(f"Cached response for key {key} with TTL {ttl}")
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [CacheManager] Cache set failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Cache set failed: {str(e)}")

    def get_cached_response(self, key: str) -> dict | None:
        """Retrieve a cached response from Redis.

        Args:
            key (str): Cache key.

        Returns:
            dict | None: Cached response data or None if not found.

        Raises:
            HTTPException: If cache retrieval fails.
        """
        try:
            cached = self.redis_client.get(key)
            if cached:
                logger.info(f"Cache hit for key {key}")
                return json.loads(cached)
            logger.info(f"Cache miss for key {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [CacheManager] Cache get failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Cache get failed: {str(e)}")

    def clear_cache(self, pattern: str = "*") -> None:
        """Clear cached responses matching a pattern.

        Args:
            pattern (str): Pattern to match cache keys (default: all keys).

        Raises:
            HTTPException: If cache clearing fails.
        """
        try:
            for key in self.redis_client.scan_iter(pattern):
                self.redis_client.delete(key)
            logger.info(f"Cleared cache for pattern {pattern}")
        except Exception as e:
            logger.error(f"Cache clear failed for pattern {pattern}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [CacheManager] Cache clear failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")
