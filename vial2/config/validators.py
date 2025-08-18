from fastapi import HTTPException
from ..config import config
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

def validate_config():
    try:
        required_env_vars = ["DATABASE_URL", "STACK_AUTH_CLIENT_ID", "STACK_AUTH_CLIENT_SECRET", "JWT_SECRET_KEY"]
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing environment variable: {var}")
        if not config.ALLOWED_ORIGINS:
            raise ValueError("ALLOWED_ORIGINS must not be empty")
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("validators", f"Config validation failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Config validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #config #validators #neon_mcp
