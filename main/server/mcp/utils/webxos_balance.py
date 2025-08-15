# main/server/mcp/utils/webxos_balance.py
import asyncio
import random
from typing import Dict
from ..utils.cache_manager import CacheManager
from ..utils.mcp_error_handler import MCPError
import logging

logger = logging.getLogger("mcp")
cache = CacheManager()

class WebXOSBalance:
    def __init__(self):
        self.base_balance = 1000.0  # Initial balance
        self.users = {}

    async def update_balance(self, user_id: str) -> float:
        try:
            if user_id not in self.users:
                self.users[user_id] = self.base_balance
            # Simulate balance fluctuation (e.g., transactions)
            fluctuation = random.uniform(-10.0, 10.0)
            self.users[user_id] += fluctuation
            await cache.set_cache(f"balance:{user_id}", {"balance": self.users[user_id], "timestamp": datetime.datetime.utcnow().isoformat()})
            logger.info(f"Updated balance for {user_id}: {self.users[user_id]}")
            return self.users[user_id]
        except Exception as e:
            logger.error(f"Balance update error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Balance update failed: {str(e)}")

    async def get_balance(self, user_id: str) -> float:
        try:
            cached = await cache.get_cache(f"balance:{user_id}")
            if cached:
                return cached["balance"]
            balance = await self.update_balance(user_id)
            return balance
        except Exception as e:
            logger.error(f"Balance retrieval error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Balance retrieval failed: {str(e)}")
