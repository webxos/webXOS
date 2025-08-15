# main/server/mcp/sync/auth_sync.py
import logging
from ..utils.cache_manager import CacheManager
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")
cache = CacheManager()

class AuthSync:
    async def sync(self, user_id: str, token: str) -> bool:
        try:
            await cache.set_cache(f"auth:{user_id}", {"token": token, "timestamp": datetime.datetime.utcnow().isoformat()})
            logger.info(f"Synced auth for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Auth sync error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Auth sync failed: {str(e)}")
