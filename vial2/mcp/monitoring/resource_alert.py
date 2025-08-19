from fastapi import APIRouter
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import psutil

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/resource_alerts")
async def get_resource_alerts():
    try:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        alert = "high" if cpu > 80 or memory > 80 else "normal"
        query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
        await neondb.execute(query, "system", "resource_alert", {"cpu": cpu, "memory": memory, "alert": alert})
        logger.info(f"Resource alert: {alert}")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"alert": alert, "cpu": cpu, "memory": memory}}}
    except Exception as e:
        error_logger.log_error("resource_alert", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Resource alert failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #resource #alert #neon_mcp
