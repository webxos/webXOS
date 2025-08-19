from fastapi import APIRouter
from langchain.adapters.monitoring import MonitoringAdapter
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import psutil
import time

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/health")
async def get_health():
    try:
        monitor = MonitoringAdapter()
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        query = "SELECT COUNT(*) FROM vial_logs WHERE created_at > NOW() - INTERVAL '1 hour'"
        active_logs = await neon_db.execute(query)
        health_metrics = monitor.assess_health({"cpu": cpu, "memory": memory, "active_logs": active_logs})
        health = "healthy" if health_metrics["status"] == "nominal" else "unhealthy"
        logger.info(f"Health check with LangChain: {health}")
        return {"jsonrpc": "2.0", "result": {"status": health, "data": health_metrics}}
    except Exception as e:
        error_logger.log_error("health_monitor", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Health monitoring failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #health #monitor #langchain #neon_mcp
