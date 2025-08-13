from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import datetime
from transformers import pipeline

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPRequest(BaseModel):
    query: str
    wallet: dict

class NLPModel:
    def __init__(self):
        self.nlp = pipeline("text-classification", model="distilbert-base-uncased")

    async def enhance_query(self, query: str, wallet: dict) -> dict:
        try:
            result = self.nlp(query)[0]
            enhanced_query = f"{query} [sentiment: {result['label']}, score: {result['score']:.2f}]"
            
            # Update wallet
            wallet["transactions"].append({
                "type": "nlp_enhancement",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            db = pymongo.MongoClient("mongodb://localhost:27017")["mcp_db"]
            db.collection("wallet").update_one(
                {"user_id": "nlp_user"},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            return {"enhanced_query": enhanced_query, "wallet": wallet}
        except Exception as e:
            logger.error(f"NLP query enhancement error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** NLP query enhancement error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

nlp_model = NLPModel()

@app.post("/api/enhance_query")
async def enhance_query(request: NLPRequest):
    return await nlp_model.enhance_query(request.query, request.wallet)
