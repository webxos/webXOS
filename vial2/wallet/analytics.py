from fastapi import HTTPException
from ..error_logging.error_log import error_logger
from ..utils.helpers import get_db_pool
import logging

logger = logging.getLogger(__name__)

async def wallet_analytics(wallet_address: str):
    try:
        async with get_db_pool() as db:
            wallet = await db.fetchrow("SELECT balance, created_at FROM wallets WHERE address=$1", wallet_address)
            if not wallet:
                raise ValueError("Wallet not found")
            transactions = await db.fetch("SELECT * FROM logs WHERE event_type='wallet_transaction' AND message LIKE $1", f"%{wallet_address}%")
            return {
                "status": "success",
                "wallet_address": wallet_address,
                "balance": wallet["balance"],
                "created_at": wallet["created_at"].isoformat(),
                "transaction_count": len(transactions)
            }
    except Exception as e:
        error_logger.log_error("wallet_analytics", f"Wallet analytics failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet analytics failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #analytics #neon_mcp
