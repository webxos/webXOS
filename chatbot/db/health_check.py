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

class HealthCheckRequest(BaseModel):
    user_id: str
    wallet: dict

class HealthCheck:
    async def check_health(self, user_id: str, wallet: dict) -> dict:
        try:
            endpoints = [
                "http://localhost:8000/api/sync_vials",
                "http://localhost:8000/api/monitor",
                "http://localhost:8000/api/collect_metrics"
            ]
            results = []
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    async with session.get(f"{endpoint}/health") as response:
                        status = await response.json() if response.status == 200 else {"status": "error", "detail": f"HTTP {response.status}"}
                        results.append({"endpoint": endpoint, "status": status["status"]})
            
            db.collection("health_logs").insert_one({
                "user_id": user_id,
                "results": results,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
            wallet["transactions"].append({
                "type": "health_check",
                "endpoints": [r["endpoint"] for r in results],
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
            logger.error(f"Health check error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Health check error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

health_check = HealthCheck()

@app.post("/api/health_check")
async def check_health(request: HealthCheckRequest):
    return await health_check.check_health(request.user_id, request.wallet)
