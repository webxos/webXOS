from fastapi import HTTPException
from ..utils.helpers import get_db_pool, log_event
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def create_escrow(wallet_address: str, amount: float, recipient: str):
    try:
        async with get_db_pool() as db:
            wallet = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", wallet_address)
            if not wallet or wallet["balance"] < amount:
                raise ValueError("Invalid wallet or insufficient balance")
            await db.execute(
                "UPDATE wallets SET balance=$1 WHERE address=$2",
                wallet["balance"] - amount, wallet_address
            )
            escrow_id = f"escrow_{wallet_address}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                "escrow", f"Escrow {escrow_id} created: {amount} from {wallet_address} to {recipient}", datetime.utcnow()
            )
            return {"status": "success", "escrow_id": escrow_id, "amount": amount}
    except Exception as e:
        error_logger.log_error("escrow", f"Escrow creation failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Escrow creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #escrow #neon_mcp
