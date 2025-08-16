from fastapi import APIRouter, Depends, HTTPException
from pymongo import MongoClient
from ...config.settings import settings
from ...security.authentication import verify_token

router = APIRouter()

@router.get("/wallet")
async def get_wallet(token: str = Depends(verify_token)):
    try:
        client = MongoClient(settings.database.url)
        db = client[settings.database.db_name]
        wallet = db.wallets.find_one({"user_id": token["sub"]})
        client.close()
        if not wallet:
            return {"balance": 0.0000, "reputation": 0}
        return {
            "balance": wallet.get("balance", 0.0000),
            "reputation": wallet.get("reputation", 0),
            "wallet_key": wallet.get("wallet_key"),
            "address": wallet.get("address")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
