from fastapi import HTTPException
from ..utils.helpers import get_db_pool, log_event
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def notify_transaction(wallet_address: str, transaction_type: str, amount: float):
    try:
        message = f"Transaction {transaction_type} of {amount} processed for wallet {wallet_address}"
        async with get_db_pool() as db:
            await log_event("wallet_notification", message, db)
        return {"status": "success", "wallet_address": wallet_address, "message": message}
    except Exception as e:
        error_logger.log_error("notifications", f"Transaction notification failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Transaction notification failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #notifications #neon_mcp
