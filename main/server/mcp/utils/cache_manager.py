# main/server/mcp/utils/cache_manager.py
import aiocache
import logging
from typing import Any, Optional
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")

class CacheManager:
    def __init__(self):
        self.cache = aiocache.SimpleMemoryCache()

    async def set_cache(self, key: str, value: Any) -> None:
        try:
            await self.cache.set(key, value, ttl=3600)  # 1-hour TTL
            logger.info(f"Set cache for {key}")
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Cache set failed: {str(e)}")

    async def get_cache(self, key: str) -> Optional[Any]:
        try:
            value = await self.cache.get(key)
            logger.info(f"Got cache for {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Cache get failed: {str(e)}")

    async def delete_cache(self, key: str) -> bool:
        try:
            result = await self.cache.delete(key)
            logger.info(f"Deleted cache for {key}")
            return result
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Cache delete failed: {str(e)}")

    async def close(self):
        await self.cache.close()
