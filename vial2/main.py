from fastapi import FastAPI
from .api.endpoints import router as endpoints_router
from .monitoring.health import router as health_router
from .monitoring.log_aggregation import router as logs_router
from .security.cors import configure_cors
from .error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
configure_cors(app)

# Include routers
app.include_router(endpoints_router)
app.include_router(health_router)
app.include_router(logs_router)

@app.on_event("startup")
async def startup_event():
    try:
        from .config import config
        config.validate()
        logger.info("Application started successfully")
    except Exception as e:
        error_logger.log_error("startup", str(e), str(e.__traceback__))
        logger.error(f"Startup failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #main #neon_mcp
