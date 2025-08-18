from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import time
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    async def execute(self, request_func):
        try:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.failures = 0
                else:
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            response = await request_func()
            self.failures = 0
            self.state = "CLOSED"
            return response
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            error_logger.log_error("circuit_breaker", str(e), str(e.__traceback__))
            logger.error(f"Circuit breaker error: {str(e)}")
            raise

# xAI Artifact Tags: #vial2 #api #circuit_breaker #neon_mcp
