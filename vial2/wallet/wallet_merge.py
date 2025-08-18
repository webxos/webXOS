from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import hashlib

logger = logging.getLogger(__name__)

async def merge_wallets(user_id: int, wallet_addresses: list):
    try:
        db = await get_db()
        total_balance = 0.0
        for address in wallet_addresses:
            wallet = await db.execute("SELECT balance FROM wallets WHERE address=$1", address)
            if not wallet:
                raise HTTPException(status_code=404, detail=f"Wallet {address} not found")
            total_balance += wallet[0]["balance"]
        merged_hash = hashlib.sha256(f"{user_id}:{':'.join(wallet_addresses)}".encode()).hexdigest()
        await db.execute(
            "INSERT INTO wallets (user_id, address, balance, hash) VALUES ($1, $2, $3, $4)",
            user_id, wallet_addresses[0], total_balance, merged_hash
        )
        await db.execute("DELETE FROM wallets WHERE address = ANY($1) AND address != $2", wallet_addresses, wallet_addresses[0])
        return {"status": "success", "merged_hash": merged_hash}
    except Exception as e:
        error_logger.log_error("wallet_merge", str(e), str(e.__traceback__))
        logger.error(f"Wallet merge failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #wallet_merge #neon_mcp
