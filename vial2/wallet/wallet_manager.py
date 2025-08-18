from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

async def import_wallet(wallet_data: dict):
    try:
        db = await get_db()
        wallet_hash = hashlib.sha256(json.dumps(wallet_data).encode()).hexdigest 
        await db.execute(
            "INSERT INTO wallets (user_id, address, balance, hash) VALUES ($1, $2, $3, $4)",
            wallet_data["user_id"], wallet_data["address"], wallet_data["balance"], wallet_hash
        )
        return {"status": "success", "wallet_hash": wallet_hash}
    except Exception as e:
        error_logger.log_error("import_wallet", str(e), str(e.__traceback__))
        logger.error(f"Wallet import failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def export_wallet(user_id: int):
    try:
        db = await get_db()
        wallet = await db.execute("SELECT * FROM wallets WHERE user_id=$1", user_id)
        return wallet
    except Exception as e:
        error_logger.log_error("export_wallet", str(e), str(e.__traceback__))
        logger.error(f"Wallet export failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #wallet_manager #neon_mcp
