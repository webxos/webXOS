from fastapi import Request, HTTPException
from ..error_logging.error_log import error_logger
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdvancedRateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    async def check_rate_limit(self, request: Request):
        try:
            client_ip = request.client.host
            current_time = time.time()
            self.requests[client_ip] = [t for t in self.requests[client_ip] if current_time - t < self.window_seconds]
            if len(self.requests[client_ip]) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(current_time)
            return True
        except Exception as e:
            error_logger.log_error("rate_limiter", f"Rate limit check failed: {str(e)}", str(e.__traceback__))
            logger.error(f"Rate limit check failed: {str(e)}")
            raise

rate_limiter = AdvancedRateLimiter()

# xAI Artifact Tags: #vial2 #api #rate_limiter #neon_mcp
