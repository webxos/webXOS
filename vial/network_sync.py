from fastapi import HTTPException
import pymongo
import redis
import logging
import datetime
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

class NetworkSync:
    def __init__(self):
        self.vials = ["nomic", "cognitallmware", "llmware", "jina_ai"]

    async def sync_vials(self, user_id: str, wallet: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sync_data = {}
            for vial in self.vials:
                # Simulate vial sync with Redis cache
                cache_key = f"vial:{vial}:{user_id}"
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    sync_data[vial] = cached_data
                else:
                    # Fetch from MongoDB
                    vial_data = db.collection("vial_data").find_one({"vial_id": vial, "user_id": user_id}) or {}
                    sync_data[vial] = vial_data.get("status", "idle")
                    redis_client.setex(cache_key, 3600, sync_data[vial])
            
            # Update wallet
            wallet["transactions"].append({
                "type": "network_sync",
                "vials": self.vials,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            # Log sync
            db.collection("sync_logs").insert_one({
                "user_id": user_id,
                "sync_data": sync_data,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
            return {"status": "synced", "data": sync_data, "wallet": wallet}
        except Exception as e:
            logger.error(f"Network sync error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Network sync error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))
