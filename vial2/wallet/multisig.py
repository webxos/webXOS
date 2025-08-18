from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import hashlib

logger = logging.getLogger(__name__)

async def create_multisig_wallet(user_id: int, addresses: list, threshold: int):
    try:
        db = await get_db()
        wallet_hash = hashlib.sha256(f"{user_id}:{':'.join(addresses)}".encode()).hexdigest()
        await db.execute(
            "INSERT INTO wallets (user_id, address, hash, balance) VALUES ($1, $2, $3, $4)",
            user_id, addresses[0], wallet_hash, 0.0
        )
        return {"wallet_address": addresses[0], "hash": wallet_hash}
    except Exception as e:
        error_logger.log_error("multisig_wallet", str(e), str(e.__traceback__))
        logger.error(f"Multisig wallet creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #multisig #neon_mcp
