from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def collect_metrics(db=Depends(get_db)):
    try:
        async with db:
            active_vials = await db.fetchval("SELECT COUNT(*) FROM vials WHERE status = 'running'")
            compute_usage = await db.fetchval("SELECT COUNT(*) FROM computes WHERE readiness = TRUE")
            recent_logs = await db.fetch("SELECT event_type, COUNT(*) as count FROM logs WHERE timestamp > NOW() - INTERVAL '1 hour' GROUP BY event_type")
        return {
            "status": "success",
            "metrics": {
                "active_vials": active_vials,
                "compute_usage": compute_usage,
                "recent_logs": {log["event_type"]: log["count"] for log in recent_logs}
            }
        }
    except Exception as e:
        error_logger.log_error("metrics", f"Metrics collection failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #monitoring #metrics #neon_mcp
