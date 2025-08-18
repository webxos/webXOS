import hashlib
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def merge_wallets(wallet_addresses: list, db):
    try:
        async with db:
            primary_wallet = wallet_addresses[0]
            total_balance = 0.0
            for address in wallet_addresses:
                wallet = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", address)
                if not wallet:
                    raise ValueError(f"Wallet {address} not found")
                total_balance += wallet["balance"]
            merged_hash = hashlib.sha256("".join(wallet_addresses).encode()).hexdigest()
            await db.execute(
                "UPDATE wallets SET balance=$1, hash=$2 WHERE address=$3",
                total_balance, merged_hash, primary_wallet
            )
            for address in wallet_addresses[1:]:
                await db.execute("DELETE FROM wallets WHERE address=$1", address)
            return {"status": "success", "primary_wallet": primary_wallet, "merged_balance": total_balance}
    except Exception as e:
        error_logger.log_error("wallet_merger", f"Wallet merge failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet merge failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #merger #neon_mcp
