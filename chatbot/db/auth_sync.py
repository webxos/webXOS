from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import jwt
import aiohttp

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class AuthSyncRequest(BaseModel):
    user_id: str
    api_key: str
    wallet: dict

class AuthSync:
    def __init__(self):
        self.secret_key = "VIAL_MCP_SECRET_2025"  # Load from .env in production

    async def sync_auth(self, user_id: str, api_key: str, wallet: dict) -> dict:
        try:
            # Verify JWT token
            decoded = jwt.decode(api_key, self.secret_key, algorithms=["HS256"])
            if decoded["user_id"] != user_id:
                raise HTTPException(status_code=401, detail="Invalid user ID in token")

            # Sync with library agents
            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/api/sync_vials", json={
                    "user_id": user_id,
                    "vials": ["1", "2", "3", "4"],
                    "wallet": wallet
                }) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail="Failed to sync vials")
                    sync_result = await response.json()

            # Update wallet
            wallet["transactions"].append({
                "type": "auth_sync",
                "user_id": user_id,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )

            # Log to MongoDB
            db.collection("auth_logs").insert_one({
                "user_id": user_id,
                "api_key": api_key[:10] + "...",  # Truncate for security
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })

            return {"status": "authenticated", "wallet": wallet, "sync_result": sync_result}
        except Exception as e:
            logger.error(f"Auth sync error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Auth sync error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

auth_sync = AuthSync()

@app.post("/api/auth_sync")
async def authenticate_and_sync(request: AuthSyncRequest):
    return await auth_sync.sync_auth(request.user_id, request.api_key, request.wallet)
