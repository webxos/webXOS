from fastapi import FastAPI
from .mcp.api.api_router import api_router
from .mcp.environment_setup import setup_environment
from .mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Vial2 MCP API")

if setup_environment():
    app.include_router(api_router, prefix="/mcp/api/vial")
    logger.info("Vial2 MCP API initialized")
else:
    logger.error("Environment setup failed, API not initialized")

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Application startup completed")
    except Exception as e:
        error_logger.log_error("startup_event", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Startup event failed: {str(e)}")

# xAI Artifact Tags: #vial2 #mcp #main #api #neon_mcp
