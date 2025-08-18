import json
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
from ..utils.helpers import get_db_pool
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def backup_wallet(wallet_address: str):
    try:
        async with get_db_pool() as db:
            wallet = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", wallet_address)
            if not wallet:
                raise ValueError("Wallet not found")
            backup_data = {
                "address": wallet["address"],
                "balance": wallet["balance"],
                "hash": wallet["hash"],
                "created_at": wallet["created_at"].isoformat(),
                "backup_timestamp": datetime.utcnow().isoformat()
            }
            with open(f"wallets/backup_{wallet_address}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.md", "w") as f:
                json.dump(backup_data, f, indent=2)
            return {"status": "success", "wallet_address": wallet_address}
    except Exception as e:
        error_logger.log_error("wallet_backup", f"Wallet backup failed for {wallet_address}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet backup failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def restore_wallet(backup_file: str):
    try:
        with open(backup_file, "r") as f:
            backup_data = json.load(f)
        async with get_db_pool() as db:
            await db.execute(
                "INSERT INTO wallets (user_id, address, balance, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3, $4) ON CONFLICT DO UPDATE SET balance=$3, hash=$4",
                backup_data["address"], backup_data["address"], backup_data["balance"], backup_data["hash"]
            )
        return {"status": "success", "wallet_address": backup_data["address"]}
    except Exception as e:
        error_logger.log_error("wallet_backup", f"Wallet restore failed for {backup_file}: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet restore failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #backup #neon_mcp
