from fastapi.middleware.cors import CORSMiddleware
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def configure_cors(app):
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://webxos.netlify.app", "http://localhost:8000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception as e:
        error_logger.log_error("cors_configuration", str(e), str(e.__traceback__))
        logger.error(f"CORS configuration failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #security #cors #neon_mcp
