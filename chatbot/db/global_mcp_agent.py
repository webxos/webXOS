from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
from typing import List, Dict

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class SyncRequest(BaseModel):
    user_id: str
    query: str
    vials: List[str]
    wallet: Dict

class GlobalMCPAgent:
    def __init__(self):
        self.vial_map = {
            "1": "Nomic",
            "2": "CogniTALLMware",
            "3": "LLMware",
            "4": "Jina AI"
        }

    async def sync_libraries(self, user_id: str, query: str, vials: List[str], wallet: Dict) -> Dict:
        try:
            results = []
            for vial_id in vials:
                if vial_id not in self.vial_map:
                    logger.warning(f"Invalid vial ID: {vial_id}")
                    continue
                response = await self._call_library_agent(vial_id, query, wallet)
                results.append({
                    "vial_id": vial_id,
                    "library": self.vial_map[vial_id],
                    "response": response["response"],
                    "wallet_update": response["wallet"]
                })
                wallet = response["wallet"]
            db.collection("sync_logs").insert_one({
                "user_id": user_id,
                "query": query,
                "vials": vials,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            wallet["transactions"].append({
                "type": "global_sync",
                "vials": vials,
                "query": query,
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
            logger.error(f"Sync error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Global sync error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

    async def _call_library_agent(self, vial_id: str, query: str, wallet: Dict) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://localhost:8000/api/library/{vial_id}", json={
                "query": query,
                "vial_id": vial_id,
                "wallet": wallet
            }) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Library agent call failed")
                return await response.json()

global_agent = GlobalMCPAgent()

@app.post("/api/sync_libraries")
async def sync_libraries(request: SyncRequest):
    return await global_agent.sync_libraries(request.user_id, request.query, request.vials, request.wallet)
