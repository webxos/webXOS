from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import aiohttp
import json

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class MetricsRequest(BaseModel):
    user_id: str
    vials: list
    wallet: dict

class PerformanceMetrics:
    async def collect_metrics(self, user_id: str, vials: list, wallet: dict) -> dict:
        try:
            with open("db/library_config.json", "r") as f:
                config = json.load(f)
            metrics = []
            async with aiohttp.ClientSession() as session:
                for vial_id in vials:
                    endpoint = config["libraries"].get(vial_id, {}).get("endpoint")
                    if not endpoint:
                        logger.warning(f"Invalid vial ID in config: {vial_id}")
                        continue
                    start_time = datetime.datetime.utcnow()
                    async with session.get(f"http://localhost:8000{endpoint}/health") as response:
                        latency = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000
                        status = await response.json() if response.status == 200 else {"status": "error", "detail": f"HTTP {response.status}"}
                        metrics.append({
                            "vial_id": vial_id,
                            "latency_ms": latency,
                            "status": status["status"],
                            "timestamp": start_time.isoformat()
                        })
            db.collection("metrics_logs").insert_one({
                "user_id": user_id,
                "vials": vials,
                "metrics": metrics,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            wallet["transactions"].append({
                "type": "metrics_collection",
                "vials": vials,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            return {"metrics": metrics, "wallet": wallet}
        except Exception as e:
            logger.error(f"Metrics collection error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Metrics collection error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

metrics_collector = PerformanceMetrics()

@app.post("/api/collect_metrics")
async def collect_metrics(request: MetricsRequest):
    return await metrics_collector.collect_metrics(request.user_id, request.vials, request.wallet)
