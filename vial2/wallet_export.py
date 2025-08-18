import json
from fastapi import HTTPException
from .error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def export_wallet(wallet_address: str, db):
    try:
        async with db:
            wallet = await db.fetchrow(
                "SELECT * FROM wallets WHERE address=$1", wallet_address
            )
            if not wallet:
                raise ValueError("Wallet not found")
            wallet_data = {
                "address": wallet["address"],
                "balance": wallet["balance"],
                "hash": wallet["hash"],
                "created_at": wallet["created_at"].isoformat()
            }
            with open(f"wallets/{wallet_address}.md", "w") as f:
                f.write(json.dumps(wallet_data, indent=2))
            return {"status": "success", "wallet_address": wallet_address}
    except Exception as e:
        error_logger.log_error("wallet_export", f"Wallet export failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet export failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def import_wallet(wallet_file: str, db):
    try:
        with open(wallet_file, "r") as f:
            wallet_data = json.load(f)
        async with db:
            await db.execute(
                "INSERT INTO wallets (user_id, address, balance, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3, $4) ON CONFLICT DO NOTHING",
                wallet_data["address"], wallet_data["address"], wallet_data["balance"], wallet_data["hash"]
            )
        return {"status": "success", "wallet_address": wallet_data["address"]}
    except Exception as e:
        error_logger.log_error("wallet_export", f"Wallet import failed for {wallet_file}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet import failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #export #neon_mcp
