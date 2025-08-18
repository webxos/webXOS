from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import hashlib

logger = logging.getLogger(__name__)

async def create_escrow_transaction(user_id: int, amount: float, recipient_address: str):
    try:
        db = await get_db()
        escrow_hash = hashlib.sha256(f"{user_id}:{amount}:{recipient_address}".encode()).hexdigest()
        await db.execute(
            "INSERT INTO wallets (user_id, address, balance, hash) VALUES ($1, $2, $3, $4)",
            user_id, recipient_address, -amount, escrow_hash
        )
        return {"status": "success", "escrow_hash": escrow_hash}
    except Exception as e:
        error_logger.log_error("escrow_transaction", str(e), str(e.__traceback__))
        logger.error(f"Escrow creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #escrow #neon_mcp
