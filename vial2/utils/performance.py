from fastapi import HTTPException
from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

async def measure_performance(endpoint: str, execution_time: float):
    try:
        async with get_db_pool() as db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                "performance", f"Endpoint {endpoint} executed in {execution_time} seconds", datetime.utcnow()
            )
        return {"status": "success", "endpoint": endpoint, "execution_time": execution_time}
    except Exception as e:
        error_logger.log_error("performance", f"Performance measurement failed for {endpoint}: {str(e)}", str(e.__traceback__))
        logger.error(f"Performance measurement failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #utils #performance #neon_mcp
