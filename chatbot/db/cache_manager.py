from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import logging
import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host="localhost", port=6379, db=0)

class CacheRequest(BaseModel):
    user_id: str
    vial_id: str
    query: str
    wallet: dict

class CacheManager:
    async def cache_response(self, user_id: str, vial_id: str, query: str, wallet: dict) -> dict:
        try:
            cache_key = f"{user_id}:{vial_id}:{query}"
            cached = redis_client.get(cache_key)
            if cached:
                return {"cached": True, "response": json.loads(cached), "wallet": wallet}

            # Fetch from library agent if not cached
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:8000/api/library/{vial_id}", json={
                    "query": query,
                    "vial_id": vial_id,
                    "wallet": wallet
                }) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail="Library agent call failed")
                    result = await response.json()

            # Cache the response
            redis_client.setex(cache_key, 3600, json.dumps(result["response"]))
            
            # Update wallet
            wallet["transactions"].append({
                "type": "cache_response",
                "vial_id": vial_id,
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
            db = mongo_client["mcp_db"]
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )

            return {"cached": False, "response": result["response"], "wallet": wallet}
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Cache error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

cache_manager = CacheManager()

@app.post("/api/cache_response")
async def cache_response(request: CacheRequest):
    return await cache_manager.cache_response(request.user_id, request.vial_id, request.query, request.wallet)
