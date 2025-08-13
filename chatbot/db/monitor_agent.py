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

class MonitorRequest(BaseModel):
    user_id: str
    vials: list
    wallet: dict

class MonitorAgent:
    async def monitor_libraries(self, user_id: str, vials: list, wallet: dict) -> dict:
        try:
            with open("db/library_config.json", "r") as f:
                config = json.load(f)
            results = []
            async with aiohttp.ClientSession() as session:
                for vial_id in vials:
                    endpoint = config["libraries"].get(vial_id, {}).get("endpoint")
                    if not endpoint:
                        logger.warning(f"Invalid vial ID in config: {vial_id}")
                        continue
                    async with session.get(f"http://localhost:8000{endpoint}/health") as response:
                        status = await response.json() if response.status == 200 else {"status": "error", "detail": f"HTTP {response.status}"}
                        results.append({"vial_id": vial_id, "health": status})
            db.collection("monitor_logs").insert_one({
                "user_id": user_id,
                "vials": vials,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            wallet["transactions"].append({
                "type": "monitor",
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
            logger.error(f"Monitor error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Monitor error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

monitor = MonitorAgent()

@app.post("/api/monitor")
async def monitor_libraries(request: MonitorRequest):
    return await monitor.monitor_libraries(request.user_id, request.vials, request.wallet)
