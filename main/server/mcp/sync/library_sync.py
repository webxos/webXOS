# main/server/mcp/sync/library_sync.py
import logging
from ..utils.cache_manager import CacheManager
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")
cache = CacheManager()

class LibrarySync:
    async def sync(self, user_id: str, data: str) -> bool:
        try:
            await cache.set_cache(f"library:{user_id}", {"data": data, "timestamp": datetime.datetime.utcnow().isoformat()})
            logger.info(f"Synced library for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Library sync error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Library sync failed: {str(e)}")
