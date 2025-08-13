from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import logging
import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigValidateRequest(BaseModel):
    user_id: str
    wallet: dict

class ConfigValidator:
    def __init__(self):
        self.required_fields = {
            "libraries": ["name", "endpoint", "model", "cache_ttl"],
            "inception_gateway": ["host", "port", "retry_attempts", "retry_delay"],
            "wallet": ["transaction_increment", "sync_interval"]
        }

    async def validate_config(self, user_id: str, wallet: dict) -> dict:
        try:
            with open("db/library_config.json", "r") as f:
                config = json.load(f)
            
            errors = []
            for section, fields in self.required_fields.items():
                if section not in config:
                    errors.append(f"Missing section: {section}")
                    continue
                for field in fields:
                    for key, value in config.get(section, {}).items():
                        if field not in value:
                            errors.append(f"Missing field {field} in {section}.{key}")
            
            status = "valid" if not errors else "invalid"
            result = {"status": status, "errors": errors}
            
            db = pymongo.MongoClient("mongodb://localhost:27017")["mcp_db"]
            db.collection("config_logs").insert_one({
                "user_id": user_id,
                "status": status,
                "errors": errors,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
            wallet["transactions"].append({
                "type": "config_validation",
                "status": status,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            return {"result": result, "wallet": wallet}
        except Exception as e:
            logger.error(f"Config validation error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Config validation error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

config_validator = ConfigValidator()

@app.post("/api/validate_config")
async def validate_config(request: ConfigValidateRequest):
    return await config_validator.validate_config(request.user_id, request.wallet)
