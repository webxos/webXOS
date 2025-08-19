from fastapi import FastAPI
from .api.api_router import api_router
from .database.neon_connection import neon_db
from .error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title="Vial2 MCP Server")

app.include_router(api_router, prefix="/mcp/api/vial")

@app.on_event("startup")
async def startup_event():
    try:
        await neon_db.connect()
        logger.info("MCP server startup completed")
    except Exception as e:
        error_logger.log_error("mcp_server_startup", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"MCP server startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await neon_db.disconnect()
        logger.info("MCP server shutdown completed")
    except Exception as e:
        error_logger.log_error("mcp_server_shutdown", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"MCP server shutdown failed: {str(e)}")

# xAI Artifact Tags: #vial2 #mcp #server #neon #mcp_server #neon_mcp
