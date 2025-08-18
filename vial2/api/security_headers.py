from fastapi import Request, Response
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def add_security_headers(request: Request, call_next):
    try:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
    except Exception as e:
        error_logger.log_error("security_headers", f"Security headers addition failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Security headers addition failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #api #security_headers #neon_mcp
