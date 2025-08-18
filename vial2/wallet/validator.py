from fastapi import HTTPException
from ..error_logging.error_log import error_logger
from ..utils.helpers import get_db_pool
import logging
import re

logger = logging.getLogger(__name__)

async def advanced_wallet_validation(wallet_data: dict):
    try:
        address = wallet_data.get("address")
        if not address or not re.match(r"^0x[a-fA-F0-9]{40}$", address):
            raise ValueError("Invalid wallet address format")
        async with get_db_pool() as db:
            existing = await db.fetchrow("SELECT * FROM wallets WHERE address=$1", address)
            if existing and wallet_data.get("action") == "create":
                raise ValueError("Wallet address already exists")
        return {"status": "success", "wallet_address": address}
    except Exception as e:
        error_logger.log_error("wallet_validator", f"Wallet validation failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Wallet validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #validator #neon_mcp
