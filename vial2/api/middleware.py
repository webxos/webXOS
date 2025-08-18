from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        try:
            client_ip = request.client.host
            current_time = time.time()
            if client_ip not in self.requests:
                self.requests[client_ip] = []
            self.requests[client_ip] = [t for t in self.requests[client_ip] if current_time - t < self.window_seconds]
            if len(self.requests[client_ip]) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(current_time)
            response = await call_next(request)
            return response
        except Exception as e:
            error_logger.log_error("middleware", f"Rate limit middleware failed: {str(e)}", str(e.__traceback__))
            logger.error(f"Rate limit middleware failed: {str(e)}")
            raise

# xAI Artifact Tags: #vial2 #api #middleware #neon_mcp
