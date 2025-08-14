import logging
import redis
import os
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class RateLimiter:
    """Manages API rate limiting for Vial MCP using Redis."""
    def __init__(self, limit: int = 100, window: int = 60):
        """Initialize RateLimiter with Redis connection and limits.

        Args:
            limit (int): Maximum requests allowed in the window.
            window (int): Time window in seconds.
        """
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.limit = limit
        self.window = window
        logger.info("RateLimiter initialized")

    def check_limit(self, wallet_id: str, endpoint: str) -> bool:
        """Check if a wallet is within the rate limit for an endpoint.

        Args:
            wallet_id (str): Wallet ID for rate limiting.
            endpoint (str): API endpoint (e.g., '/api/notes/add').

        Returns:
            bool: True if within limit, False otherwise.

        Raises:
            HTTPException: If rate limit check fails.
        """
        try:
            key = f"rate:{wallet_id}:{endpoint}"
            current = self.redis_client.get(key)
            if current is None:
                self.redis_client.setex(key, self.window, 1)
                return True
            elif int(current) < self.limit:
                self.redis_client.incr(key)
                return True
            else:
                logger.warning(f"Rate limit exceeded for wallet {wallet_id} on {endpoint}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception as e:
            logger.error(f"Rate limit check failed for {key}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [RateLimiter] Rate limit check failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Rate limit check failed: {str(e)}")
