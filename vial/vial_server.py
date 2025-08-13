from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import os
from typing import Dict, Any

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]

class VialRequest(BaseModel):
    user_id: str
    vial_id: str
    command: str
    wallet: Dict[str, Any]

class VialServer:
    def __init__(self):
        self.vials = ["nomic", "cognitallmware", "llmware", "jina_ai"]

    async def process_vial_command(self, user_id: str, vial_id: str, command: str, wallet: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if vial_id not in self.vials:
                raise ValueError(f"Invalid vial_id: {vial_id}")
            
            # Simulate command processing
            result = f"Processed {command} for {vial_id}"
            
            # Update wallet
            wallet["transactions"].append({
                "type": "vial_command",
                "vial_id": vial_id,
                "command": command,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            # Log command
            db.collection("vial_logs").insert_one({
                "user_id": user_id,
                "vial_id": vial_id,
                "command": command,
                "result": result,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
            return {"status": result, "vial_id": vial_id, "wallet": wallet}
        except Exception as e:
            logger.error(f"Vial command error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Vial command error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

vial_server = VialServer()

@app.post("/api/manage_vial")
async def manage_vial(request: VialRequest):
    return await vial_server.process_vial_command(request.user_id, request.vial_id, request.command, request.wallet)
