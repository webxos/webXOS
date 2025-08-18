from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    async def get_cached_data(self, key: str):
        try:
            return self.cache.get(key)
        except Exception as e:
            error_logger.log_error("cache", f"Cache get failed for {key}: {str(e)}", str(e.__traceback__))
            logger.error(f"Cache get failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def set_cached_data(self, key: str, value):
        try:
            self.cache[key] = value
            return {"status": "success", "key": key}
        except Exception as e:
            error_logger.log_error("cache", f"Cache set failed for {key}: {str(e)}", str(e.__traceback__))
            logger.error(f"Cache set failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

cache_manager = CacheManager()

# xAI Artifact Tags: #vial2 #utils #cache #neon_mcp
