from fastapi import HTTPException
from ..utils.helpers import get_db_pool, log_event
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def process_transaction(wallet_address: str, amount: float, transaction_type: str):
    try:
        async with get_db_pool() as db:
            wallet = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", wallet_address)
            if not wallet:
                raise ValueError("Wallet not found")
            new_balance = wallet["balance"] + (amount if transaction_type == "credit" else -amount)
            if new_balance < 0:
                raise ValueError("Insufficient balance")
            await db.execute(
                "UPDATE wallets SET balance=$1 WHERE address=$2",
                new_balance, wallet_address
            )
            await log_event("wallet_transaction", f"Transaction {transaction_type} of {amount} for {wallet_address}", db)
            return {"status": "success", "wallet_address": wallet_address, "new_balance": new_balance}
    except Exception as e:
        error_logger.log_error("transactions", f"Transaction failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Transaction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #transactions #neon_mcp
