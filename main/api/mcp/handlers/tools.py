from fastapi import Depends
from utils.auth import verify_token

async def get_wallet_info(token: str = Depends(verify_token)):
    return {"user_id": token["sub"], "balance": 0.0, "currency": "USD"}

async def generate_credentials(token: str = Depends(verify_token)):
    return {"key": "mock_key", "secret": "mock_secret"}
