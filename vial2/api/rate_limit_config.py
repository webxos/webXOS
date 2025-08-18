from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging
import json
import os

logger = logging.getLogger(__name__)

class RateLimitConfig:
    def __init__(self, config_file: str = "rate_limits.json"):
        self.config_file = config_file
        self.limits = self.load_config()

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    return json.load(f)
            return {"default": {"max_requests": 100, "window_seconds": 60}}
        except Exception as e:
            error_logger.log_error("rate_limit_config", f"Failed to load rate limit config: {str(e)}", str(e.__traceback__))
            logger.error(f"Failed to load rate limit config: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def get_limit(self, endpoint: str):
        try:
            return self.limits.get(endpoint, self.limits["default"])
        except Exception as e:
            error_logger.log_error("rate_limit_config", f"Failed to get rate limit for {endpoint}: {str(e)}", str(e.__traceback__))
            logger.error(f"Failed to get rate limit: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

rate_limit_config = RateLimitConfig()

# xAI Artifact Tags: #vial2 #api #rate_limit_config #neon_mcp
