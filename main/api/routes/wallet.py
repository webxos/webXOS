from fastapi import APIRouter, Depends, HTTPException
from pymongo import MongoClient
from ...config.settings import settings
from ...security.authentication import verify_token
import uuid

router = APIRouter()

@router.get("/wallet")
async def get_wallet(token: str = Depends(verify_token)):
    try:
        client = MongoClient(settings.database.url)
        db = client[settings.database.db_name]
        wallet = db.wallets.find_one({"user_id": token["sub"]})
        client.close()
        if not wallet:
            new_wallet = {
                "user_id": token["sub"],
                "balance": 0.0000,
                "reputation": 0,
                "wallet_key": str(uuid.uuid4()),
                "address": str(uuid.uuid4())
            }
            db.wallets.insert_one(new_wallet)
            return {
                "balance": new_wallet["balance"],
                "reputation": new_wallet["reputation"],
                "wallet_key": new_wallet["wallet_key"],
                "address": new_wallet["address"]
            }
        return {
            "balance": wallet.get("balance", 0.0000),
            "reputation": wallet.get("reputation", 0),
            "wallet_key": wallet.get("wallet_key"),
            "address": wallet.get("address")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
