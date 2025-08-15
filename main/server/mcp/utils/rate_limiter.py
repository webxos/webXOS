# main/server/mcp/utils/rate_limiter.py
from typing import Optional
import redis.asyncio as redis
from ..utils.mcp_error_handler import MCPError
import time
import os

class RateLimiter:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.requests_per_window = 60
        self.window_seconds = 60
        self.burst_size = 10

    async def check_rate_limit(self, user_id: str, endpoint: str) -> None:
        try:
            key = f"rate_limit:{user_id}:{endpoint}"
            current_time = int(time.time())
            window_start = current_time - self.window_seconds

            # Clean up old requests
            await self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            request_count = await self.redis_client.zcard(key)
            if request_count >= self.requests_per_window:
                raise MCPError(code=-32029, message="Rate limit exceeded")

            # Add new request timestamp
            await self.redis_client.zadd(key, {str(current_time): current_time})
            await self.redis_client.expire(key, self.window_seconds)

            # Check burst limit
            recent_requests = await self.redis_client.zrangebyscore(key, current_time - 1, current_time)
            if len(recent_requests) > self.burst_size:
                raise MCPError(code=-32029, message="Burst limit exceeded")
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Rate limiting failed: {str(e)}")

    async def clear_rate_limit(self, user_id: str, endpoint: Optional[str] = None) -> None:
        try:
            key_pattern = f"rate_limit:{user_id}:{endpoint or '*'}"
            async for key in self.redis_client.scan_iter(key_pattern):
                await self.redis_client.delete(key)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to clear rate limit: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()
