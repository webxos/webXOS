from fastapi import APIRouter
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/alerts")
async def get_alerts():
    try:
        query = "SELECT COUNT(*) FROM vial_logs WHERE event_type = 'error' AND created_at > NOW() - INTERVAL '1 hour'"
        count = await neon_db.execute(query)
        alerts = {"error_count": count, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if int(count) > 5:
            logger.warning(f"High error count detected: {count}")
        logger.info("Alert data retrieved")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": alerts}}
    except Exception as e:
        error_logger.log_error("alert_manager", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Alert retrieval failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #alert #manager #neon_mcp
