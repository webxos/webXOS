from fastapi import HTTPException
from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def collect_telemetry(metric_name: str, value: float):
    try:
        async with get_db_pool() as db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                "telemetry", f"Metric {metric_name}: {value}", datetime.utcnow()
            )
        return {"status": "success", "metric_name": metric_name, "value": value}
    except Exception as e:
        error_logger.log_error("telemetry", f"Telemetry collection failed for {metric_name}: {str(e)}", str(e.__traceback__))
        logger.error(f"Telemetry collection failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #utils #telemetry #neon_mcp
