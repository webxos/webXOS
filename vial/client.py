from fastapi import FastAPI, HTTPException
import aiohttp
import logging
import datetime
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientRequest(BaseModel):
    user_id: str
    endpoint: str
    payload: dict
    wallet: dict

class MCPClient:
    async def send_request(self, user_id: str, endpoint: str, payload: dict, wallet: dict) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{os.getenv('INCEPTION_GATEWAY_HOST', 'localhost')}:{os.getenv('INCEPTION_GATEWAY_PORT', '8000')}{endpoint}", json=payload) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail="Request failed")
                    result = await response.json()
            
            # Update wallet
            db = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))["mcp_db"]
            wallet["transactions"].append({
                "type": "client_request",
                "endpoint": endpoint,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            return {"result": result, "wallet": wallet}
        except Exception as e:
            logger.error(f"Client request error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Client request error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

mcp_client = MCPClient()

@app.post("/api/client_request")
async def send_client_request(request: ClientRequest):
    return await mcp_client.send_request(request.user_id, request.endpoint, request.payload, request.wallet)
