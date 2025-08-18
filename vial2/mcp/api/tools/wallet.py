import hashlib
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def validate_wallet(wallet_data: dict, db):
    try:
        address = wallet_data.get("address")
        if not address or not address.startswith("0x") or len(address) != 42:
            raise ValueError("Invalid wallet address")
        
        wallet_hash = hashlib.sha256(address.encode()).hexdigest()
        async with db:
            await db.execute(
                "INSERT INTO wallets (user_id, address, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3) ON CONFLICT DO NOTHING",
                address, address, wallet_hash
            )
        return {"wallet_address": address, "hash": wallet_hash}
    except Exception as e:
        logger.error(f"Wallet validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #wallet #webxos #neon_mcp
