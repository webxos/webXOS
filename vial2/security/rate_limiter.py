from fastapi import Request, HTTPException
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def check_limit(self, request: Request):
        try:
            client_ip = request.client.host
            current_time = time.time()
            if client_ip not in self.requests:
                self.requests[client_ip] = []
            self.requests[client_ip] = [t for t in self.requests[client_ip] if current_time - t < self.window_seconds]
            if len(self.requests[client_ip]) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(current_time)
        except Exception as e:
            error_logger.log_error("rate_limiter", str(e), str(e.__traceback__))
            logger.error(f"Rate limiting failed: {str(e)}")
            raise

rate_limiter = RateLimiter()

# xAI Artifact Tags: #vial2 #security #rate_limiter #neon_mcp
