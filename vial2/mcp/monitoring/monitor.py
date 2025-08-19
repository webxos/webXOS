from fastapi import APIRouter
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/monitor")
async def get_monitor_data():
    try:
        query = "SELECT COUNT(*) FROM vial_logs WHERE created_at > NOW() - INTERVAL '1 hour'"
        count = await neon_db.execute(query)
        metrics = {
            "active_logs": count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        logger.info("Monitor data retrieved")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": metrics}}
    except Exception as e:
        error_logger.log_error("monitor_data", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Monitor data fetch failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #neon #monitor #neon_mcp
