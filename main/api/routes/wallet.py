from fastapi import APIRouter, Depends
from utils.auth import verify_token
from providers.redis_config import redis_client

router = APIRouter()

@router.get("/wallet")
async def get_wallet_info(token: str = Depends(verify_token)):
    wallet_data = redis_client.get(f"wallet:{token['sub']}")
    if wallet_data:
        return wallet_data
    return {"user_id": token["sub"], "balance": 0.0, "currency": "USD"}
