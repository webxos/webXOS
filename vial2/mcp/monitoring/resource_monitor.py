from fastapi import APIRouter
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import psutil
import time

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/resources")
async def get_resources():
    try:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
        await neondb.execute(query, "system", "resource_monitor", {"cpu": cpu, "memory": memory})
        logger.info("Resource data logged")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"cpu": cpu, "memory": memory, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}}
    except Exception as e:
        error_logger.log_error("resource_monitor", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Resource monitoring failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #resource #monitor #neon_mcp
