from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import aiohttp

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class LibrarySyncRequest(BaseModel):
    user_id: str
    vials: list
    wallet: dict

class LibrarySync:
    async def sync_vials(self, user_id: str, vials: list, wallet: dict) -> dict:
        try:
            results = []
            for vial_id in vials:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:8000/api/library/{vial_id}/status") as response:
                        if response.status != 200:
                            logger.warning(f"Failed to sync vial {vial_id}: HTTP {response.status}")
                            continue
                        status = await response.json()
                        results.append({"vial_id": vial_id, "status": status})
            db.collection("sync_logs").insert_one({
                "user_id": user_id,
                "vials": vials,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            wallet["transactions"].append({
                "type": "library_sync",
                "vials": vials,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            return {"results": results, "wallet": wallet}
        except Exception as e:
            logger.error(f"Library sync error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Library sync error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

library_sync = LibrarySync()

@app.post("/api/sync_vials")
async def sync_vials(request: LibrarySyncRequest):
    return await library_sync.sync_vials(request.user_id, request.vials, request.wallet)
