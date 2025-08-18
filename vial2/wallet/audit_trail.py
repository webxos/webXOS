from fastapi import HTTPException
from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def generate_audit_trail(wallet_address: str):
    try:
        async with get_db_pool() as db:
            transactions = await db.fetch(
                "SELECT * FROM logs WHERE event_type='wallet_transaction' AND message LIKE $1 ORDER BY timestamp DESC",
                f"%{wallet_address}%"
            )
            return {
                "status": "success",
                "wallet_address": wallet_address,
                "audit_trail": [
                    {"event": log["event_type"], "message": log["message"], "timestamp": log["timestamp"].isoformat()}
                    for log in transactions
                ]
            }
    except Exception as e:
        error_logger.log_error("audit_trail", f"Audit trail generation failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Audit trail generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #audit_trail #neon_mcp
