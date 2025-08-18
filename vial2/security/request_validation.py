from fastapi import HTTPException, Request
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def validate_request(request: Request):
    try:
        if not request.headers.get("Authorization"):
            raise HTTPException(status_code=401, detail="Missing authorization header")
        if not request.url.path.startswith("/mcp/api"):
            raise HTTPException(status_code=403, detail="Invalid API path")
        return True
    except Exception as e:
        error_logger.log_error("request_validation", str(e), str(e.__traceback__))
        logger.error(f"Request validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #security #request_validation #neon_mcp
