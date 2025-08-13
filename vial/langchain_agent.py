from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import logging
import datetime
import os
from langchain import LLMChain, PromptTemplate
from transformers import pipeline

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]

class LangChainRequest(BaseModel):
    user_id: str
    query: str
    wallet: dict

class LangChainAgent:
    def __init__(self):
        self.nlp = pipeline("text-generation", model="distilgpt2")
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Process this query for WebXOS: {query}"
        )
        self.chain = LLMChain(llm=self.nlp, prompt=self.prompt_template)

    async def process_query(self, user_id: str, query: str, wallet: dict) -> dict:
        try:
            result = self.chain.run(query=query)
            
            # Log query
            db.collection("langchain_logs").insert_one({
                "user_id": user_id,
                "query": query,
                "response": result,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "wallet": wallet
            })
            
            # Update wallet
            wallet["transactions"].append({
                "type": "langchain_query",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            return {"response": result, "wallet": wallet}
        except Exception as e:
            logger.error(f"LangChain query error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** LangChain query error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

langchain_agent = LangChainAgent()

@app.post("/api/langchain_query")
async def process_langchain_query(request: LangChainRequest):
    return await langchain_agent.process_query(request.user_id, request.query, request.wallet)
