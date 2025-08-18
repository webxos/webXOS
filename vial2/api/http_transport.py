from fastapi import FastAPI, Request, HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def configure_http_transport(app: FastAPI):
    try:
        @app.middleware("http")
        async def check_content_type(request: Request, call_next):
            if request.method == "POST" and request.headers.get("content-type") != "application/json":
                error_logger.log_error("http_transport", "Invalid content type", "")
                logger.error("Invalid content type")
                raise HTTPException(status_code=415, detail="Content-Type must be application/json")
            response = await call_next(request)
            response.headers["Content-Type"] = "application/json"
            return response
    except Exception as e:
        error_logger.log_error("http_transport_setup", str(e), str(e.__traceback__))
        logger.error(f"HTTP transport setup failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #api #http_transport #neon_mcp
