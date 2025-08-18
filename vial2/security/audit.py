from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def audit_security_events():
    try:
        async with get_db_pool() as db:
            suspicious_logs = await db.fetch(
                "SELECT * FROM logs WHERE event_type IN ('auth_failure', 'wallet_error') AND timestamp > NOW() - INTERVAL '24 hours'"
            )
            return {
                "status": "success",
                "audit_report": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "suspicious_events": [dict(log) for log in suspicious_logs]
                }
            }
    except Exception as e:
        error_logger.log_error("audit", f"Security audit failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Security audit failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #security #audit #neon_mcp
