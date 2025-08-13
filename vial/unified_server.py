from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from pydantic import BaseModel
import logging
import re
import hashlib
import datetime
import aiofiles
import time
from vial.webxos_wallet import WebXOSWallet

app = FastAPI()

# MongoDB setup with retry
def connect_mongo():
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            client = MongoClient('mongodb://mongo:27017', serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return client
        except Exception as e:
            logging.error(f"MongoDB connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}")

mongo_client = connect_mongo()
db = mongo_client['mcp_db']

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models
class AuthRequest(BaseModel):
    userId: str

class QueryRequest(BaseModel):
    query: str
    timestamp: str

class WalletRequest(BaseModel):
    transaction: dict
    wallet: dict

class EnhanceQueryRequest(BaseModel):
    query: str
    apiKey: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    user = db.users.find_one({"apiKey": api_key})
    if not user:
        logger.error(f"Invalid API key: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    return user

@app.get("/api/health")
async def health():
    try:
        db.command("ping")
        logger.info("Health check passed")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        db.errors.insert_one({
            "error": f"Health check failed: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")

@app.post("/api/auth")
async def auth(auth_request: AuthRequest):
    try:
        api_key = hashlib.sha256(f"{auth_request.userId}{datetime.datetime.utcnow().isoformat()}".encode()).hexdigest()
        user = {
            "userId": auth_request.userId,
            "apiKey": api_key,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        db.users.update_one({"userId": auth_request.userId}, {"$set": user}, upsert=True)
        logger.info(f"Authenticated user: {auth_request.userId}")
        db.queries.insert_one({
            "query": f"Authentication for {auth_request.userId}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": auth_request.userId
        })
        return {"apiKey": api_key}
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        db.errors.insert_one({
            "error": f"Auth error: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vials", dependencies=[Depends(verify_token)])
async def get_vials():
    try:
        vials = list(db.vials.find({}, {"_id": 0}))
        agents = {vial["id"]: {k: v for k, v in vial.items() if k != "id"} for vial in vials}
        logger.info(f"Retrieved {len(agents)} vials")
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to load vials: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to load vials: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wallet", dependencies=[Depends(verify_token)])
async def update_wallet(wallet_request: WalletRequest, user: dict = Depends(verify_token)):
    try:
        wallet = WebXOSWallet()
        db.wallet.update_one(
            {"userId": user["userId"]},
            {"$push": {"transactions": wallet_request.transaction}, "$inc": {"webxos": 0.0001}},
            upsert=True
        )
        wallet.update_balance(user["userId"], wallet_request.transaction.get("amount", 0.0))
        logger.info(f"Updated wallet for user: {user['userId']}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update wallet: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to update wallet: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wallet/cashout", dependencies=[Depends(verify_token)])
async def cashout_wallet(request: WalletRequest, user: dict = Depends(verify_token)):
    try:
        wallet = WebXOSWallet()
        wallet.cashout(user["userId"], request.wallet["target_address"], request.transaction["amount"])
        db.wallet.update_one(
            {"userId": user["userId"]},
            {"$push": {"transactions": request.transaction}, "$inc": {"webxos": -request.transaction["amount"]}},
            upsert=True
        )
        logger.info(f"Cashed out {request.transaction['amount']} $WEBXOS for user: {user['userId']}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Cashout error: {str(e)}")
        db.errors.insert_one({
            "error": f"Cashout error: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import", dependencies=[Depends(verify_token)])
async def import_vials(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    try:
        async with aiofiles.open(f"/tmp/{file.filename}", 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        with open(f"/tmp/{file.filename}", 'r') as f:
            text = f.read()
        wallet = WebXOSWallet()
        if not wallet.validate_export(text):
            logger.error(f"Invalid wallet export format: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid wallet export format")
        agent_sections = text.split('# Vial Agent: ')[1:]
        if len(agent_sections) != 4:
            logger.error(f"Invalid number of vials: expected 4, found {len(agent_sections)}")
            raise HTTPException(status_code=400, detail=f"Expected 4 vials, found {len(agent_sections)}")
        agents = {}
        for section in agent_sections:
            lines = section.split('\n')
            id_line = lines[0].strip()
            id = id_line.split(': ')[1] if ': ' in id_line else ''
            status = next((line.split('Status: ')[1] for line in lines if line.startswith('Status: ')), '')
            wallet_balance = float(next((line.split('Wallet Balance: ')[1].split(' ')[0] for line in lines if line.startswith('Wallet Balance: ')), 0))
            wallet_address = next((line.split('Wallet Address: ')[1] for line in lines if line.startswith('Wallet Address: ')), '')
            wallet_hash = next((line.split('Wallet Hash: ')[1] for line in lines if line.startswith('Wallet Hash: ')), '')
            script_start = next(i for i, line in enumerate(lines) if line.startswith('```python'))
            script_end = next(i for i, line in enumerate(lines[script_start:]) if line.startswith('```') and i > 0) + script_start
            script = '\n'.join(lines[script_start + 1:script_end])
            agents[id] = {"status": status, "wallet_balance": wallet_balance, "wallet_address": wallet_address, "wallet_hash": wallet_hash, "script": script}
            wallet.update_balance(id, wallet_balance)
        db.vials.delete_many({})
        db.vials.insert_many([{"id": id, **agent} for id, agent in agents.items()])
        logger.info(f"Imported {len(agents)} vials from {file.filename} for user: {user['userId']}")
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to import vials: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to import vials: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    try:
        db.command("ping")
        db.users.create_index("apiKey")
        db.wallet.create_index("userId")
        logger.info("Connected to MongoDB with indexes")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to connect to MongoDB: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise Exception(f"Failed to connect to MongoDB: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    mongo_client.close()
    logger.info("Disconnected from MongoDB")
