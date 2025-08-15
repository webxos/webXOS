# main/server/mcp/utils/rate_limiter.py
from typing import Optional
import redis.asyncio as redis
from fastapi import Request, HTTPException
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
import os
import logging
import time
import asyncio

logger = logging.getLogger("mcp")

class RateLimiter:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.metrics = PerformanceMetrics()
        self.rate_limit = int(os.getenv("RATE_LIMIT_REQUESTS", 100))  # Requests per window
        self.window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))  # 1 minute

    async def check_rate_limit(self, request: Request) -> None:
        try:
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}"
            current_time = time.time()
            
            # Use Redis pipeline for atomic operations
            async with self.redis_client.pipeline() as pipe:
                # Get current count and window start
                pipe.get(key)
                pipe.ttl(key)
                count, ttl = await pipe.execute()
                
                if count is None:
                    # Initialize new window
                    await self.redis_client.setex(key, self.window_seconds, 1)
                    self.metrics.requests_total.labels(endpoint="rate_limit").inc()
                    logger.debug(f"New rate limit window for {client_ip}")
                    return
                
                count = int(count)
                if count >= self.rate_limit:
                    retry_after = max(0, ttl)
                    logger.warning(f"Rate limit exceeded for {client_ip}, retry after {retry_after}s")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Try again in {retry_after} seconds."
                    )
                
                # Increment count
                await self.redis_client.incr(key)
                self.metrics.requests_total.labels(endpoint="rate_limit").inc()
                logger.debug(f"Rate limit check passed for {client_ip}: {count + 1}/{self.rate_limit}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to check rate limit: {str(e)}")

    async def reset_rate_limit(self, client_ip: str) -> None:
        try:
            key = f"rate_limit:{client_ip}"
            await self.redis_client.delete(key)
            logger.info(f"Reset rate limit for {client_ip}")
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to reset rate limit: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()
