from fastapi import HTTPException
from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def monitor_api_rates(endpoint: str, client_ip: str):
    try:
        async with get_db_pool() as db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                "api_rate", f"Request to {endpoint} from {client_ip}", datetime.utcnow()
            )
            rate_count = await db.fetchval(
                "SELECT COUNT(*) FROM logs WHERE event_type='api_rate' AND message LIKE $1 AND timestamp > NOW() - INTERVAL '1 hour'",
                f"%{client_ip}%"
            )
            return {"status": "success", "endpoint": endpoint, "client_ip": client_ip, "request_count": rate_count}
    except Exception as e:
        error_logger.log_error("rate_monitor", f"API rate monitoring failed for {endpoint}: {str(e)}", str(e.__traceback__))
        logger.error(f"API rate monitoring failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #utils #rate_monitor #neon_mcp
