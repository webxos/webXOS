from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import json
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]

class VialRequest(BaseModel):
    user_id: str
    vial_id: str
    command: str
    wallet: dict

class VialManager:
    async def manage_vial(self, user_id: str, vial_id: str, command: str, wallet: dict) -> dict:
        try:
            with open("db/library_config.json", "r") as f:
                config = json.load(f)
            
            if vial_id not in config["libraries"]:
                raise HTTPException(status_code=400, detail=f"Invalid vial ID: {vial_id}")
            
            # Log command
            db.collection("vial_logs").insert_one({
                "user_id": user_id,
                "vial_id": vial_id,
                "command": command,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
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
            
            return {"status": "command processed", "vial_id": vial_id, "wallet": wallet}
        except Exception as e:
            logger.error(f"Vial management error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Vial management error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

vial_manager = VialManager()

@app.post("/api/manage_vial")
async def manage_vial(request: VialRequest):
    return await vial_manager.manage_vial(request.user_id, request.vial_id, request.command, request.wallet)
