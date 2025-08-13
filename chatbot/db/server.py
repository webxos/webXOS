from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import json

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["mcp_db"]

class QueryRequest(BaseModel):
    user_id: str
    query: str
    wallet: dict

@app.get("/health")
async def health_check():
    try:
        mongo_client.server_info()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Health check error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def process_query(request: QueryRequest):
    try:
        with open("db/library_config.json", "r") as f:
            config = json.load(f)
        
        # Log query
        db.collection("query_logs").insert_one({
            "user_id": request.user_id,
            "query": request.query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "wallet": request.wallet
        })
        
        # Update wallet
        request.wallet["transactions"].append({
            "type": "db_query",
            "query": request.query,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        request.wallet["webxos"] = request.wallet.get("webxos", 0.0) + 0.0001
        db.collection("wallet").update_one(
            {"user_id": request.user_id},
            {"$set": {"wallet": request.wallet}, "$push": {"transactions": request.wallet["transactions"][-1]}},
            upsert=True
        )
        
        return {"status": "processed", "wallet": request.wallet}
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Query processing error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
