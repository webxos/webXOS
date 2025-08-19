from langchain.cache import InMemoryCache
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.cache = InMemoryCache()

    def get_cache(self, key: str):
        try:
            value = self.cache.get(key)
            logger.info(f"Retrieved cache for key: {key}")
            return value
        except Exception as e:
            error_logger.log_error("cache_get", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Cache retrieval failed: {str(e)}")
            raise

    def set_cache(self, key: str, value):
        try:
            self.cache.set(key, value)
            logger.info(f"Set cache for key: {key}")
        except Exception as e:
            error_logger.log_error("cache_set", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Cache set failed: {str(e)}")
            raise

cache_manager = CacheManager()

# xAI Artifact Tags: #vial2 #mcp #langchain #cache #manager #neon_mcp
