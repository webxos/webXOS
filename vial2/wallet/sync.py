from fastapi import HTTPException
from ..utils.helpers import get_db_pool, log_event
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def sync_wallet(wallet_address: str):
    try:
        async with get_db_pool() as db:
            wallet = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", wallet_address)
            if not wallet:
                raise ValueError("Wallet not found")
            await log_event("wallet_sync", f"Synchronized wallet {wallet_address}", db)
            return {"status": "success", "wallet_address": wallet_address, "balance": wallet["balance"]}
    except Exception as e:
        error_logger.log_error("wallet_sync", f"Wallet sync failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet sync failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #sync #neon_mcp
