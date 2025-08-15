# main/server/mcp/utils/cache_manager.py
from typing import Any, Optional
import redis.asyncio as redis
from ..utils.mcp_error_handler import MCPError
import json
import os

class CacheManager:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.default_ttl = 3600  # 1 hour

    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl or self.default_ttl, serialized)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to set cache: {str(e)}")

    async def get_cache(self, key: str) -> Optional[Any]:
        try:
            serialized = await self.redis_client.get(key)
            if serialized is None:
                return None
            return json.loads(serialized)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to get cache: {str(e)}")

    async def delete_cache(self, key: str) -> None:
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to delete cache: {str(e)}")

    async def clear_cache(self) -> None:
        try:
            await self.redis_client.flushdb()
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to clear cache: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()
