# main/server/mcp/utils/rate_limiter.py
import time
from typing import Dict
from ..utils.mcp_error_handler import MCPError

class RateLimiter:
    def __init__(self, limit: int = 10, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests: Dict[str, list] = {}

    async def check(self, user_id: str) -> bool:
        current_time = time.time()
        if user_id not in self.requests:
            self.requests[user_id] = []
        self.requests[user_id] = [t for t in self.requests[user_id] if current_time - t < self.window]
        if len(self.requests[user_id]) >= self.limit:
            raise MCPError(code=-32604, message="Rate limit exceeded")
        self.requests[user_id].append(current_time)
        return True
