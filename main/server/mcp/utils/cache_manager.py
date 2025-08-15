# main/server/mcp/utils/cache_manager.py
from typing import Any, Optional
import redis.asyncio as redis
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
import os
import logging
import json
import asyncio

logger = logging.getLogger("mcp")

class CacheManager:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.metrics = PerformanceMetrics()
        self.cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # Default 1 hour

    @self.metrics.track_request("set_cache")
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            if not key:
                raise MCPError(code=-32602, message="Cache key is required")
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl or self.cache_ttl, serialized)
            logger.info(f"Set cache for key {key}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to set cache: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to set cache: {str(e)}")

    @self.metrics.track_request("get_cache")
    async def get_cache(self, key: str) -> Optional[Any]:
        try:
            if not key:
                raise MCPError(code=-32602, message="Cache key is required")
            serialized = await self.redis_client.get(key)
            if serialized is None:
                logger.info(f"Cache miss for key {key}")
                return None
            value = json.loads(serialized)
            logger.info(f"Cache hit for key {key}")
            return value
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to get cache: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to get cache: {str(e)}")

    @self.metrics.track_request("delete_cache")
    async def delete_cache(self, key: str) -> None:
        try:
            if not key:
                raise MCPError(code=-32602, message="Cache key is required")
            await self.redis_client.delete(key)
            logger.info(f"Deleted cache for key {key}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to delete cache: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to delete cache: {str(e)}")

    @self.metrics.track_request("clear_cache")
    async def clear_cache(self, pattern: str = "*") -> None:
        try:
            keys = [key async for key in self.redis_client.scan_iter(match=pattern)]
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys matching {pattern}")
            else:
                logger.info(f"No cache keys found matching {pattern}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to clear cache: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()
