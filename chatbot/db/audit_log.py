from fastapi import FastAPI
from pydantic import BaseModel
import pymongo
import logging
import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class AuditRequest(BaseModel):
    user_id: str
    action: str
    wallet: dict

class AuditLog:
    async def log_action(self, user_id: str, action: str, wallet: dict) -> dict:
        try:
            audit_entry = {
                "user_id": user_id,
                "action": action,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            }
            db.collection("audit_logs").insert_one(audit_entry)
            
            # Update wallet
            wallet["transactions"].append({
                "type": "audit_log",
                "action": action,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            return {"status": "logged", "wallet": wallet}
        except Exception as e:
            logger.error(f"Audit log error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Audit log error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

audit_log = AuditLog()

@app.post("/api/audit_log")
async def log_action(request: AuditRequest):
    return await audit_log.log_action(request.user_id, request.action, request.wallet)
