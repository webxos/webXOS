from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from pydantic import BaseModel
import logging
import re
import hashlib
import datetime
import aiofiles

app = FastAPI()

# MongoDB setup
mongo_client = MongoClient('mongodb://mongo:27017')
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
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    return user

@app.get("/api/health")
async def health():
    try:
        db.command("ping")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
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
        return {"apiKey": api_key}
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vials", dependencies=[Depends(verify_token)])
async def get_vials():
    try:
        vials = list(db.vials.find({}, {"_id": 0}))
        agents = {vial["id"]: {k: v for k, v in vial.items() if k != "id"} for vial in vials}
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to load vials: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log_query", dependencies=[Depends(verify_token)])
async def log_query(query_request: QueryRequest):
    try:
        db.queries.insert_one({
            "query": query_request.query,
            "timestamp": query_request.timestamp,
            "userId": verify_token(Depends(security)).userId
        })
        logger.info(f"Logged query: {query_request.query}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to log query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wallet", dependencies=[Depends(verify_token)])
async def update_wallet(wallet_request: WalletRequest):
    try:
        db.wallet.update_one(
            {"userId": verify_token(Depends(security)).userId},
            {"$push": {"transactions": wallet_request.transaction}, "$inc": {"webxos": 0.0001}},
            upsert=True
        )
        logger.info(f"Updated wallet for user: {verify_token(Depends(security)).userId}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update wallet: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhance_query", dependencies=[Depends(verify_token)])
async def enhance_query(request: EnhanceQueryRequest):
    try:
        # Call NLP model for query enhancement
        from nlp_model import enhance_query
        enhanced_query = enhance_query(request.query)
        db.queries.insert_one({
            "query": request.query,
            "enhanced_query": enhanced_query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": request.apiKey
        })
        logger.info(f"Enhanced query: {request.query} -> {enhanced_query}")
        return {"enhanced_query": enhanced_query}
    except Exception as e:
        logger.error(f"Failed to enhance query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import", dependencies=[Depends(verify_token)])
async def import_vials(file: UploadFile = File(...)):
    try:
        async with aiofiles.open(f"/tmp/{file.filename}", 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        with open(f"/tmp/{file.filename}", 'r') as f:
            text = f.read()
        agent_sections = text.split('## Agent ')[1:]
        if len(agent_sections) != 4:
            raise HTTPException(status_code=400, detail=f"Expected 4 vials, found {len(agent_sections)}")
        agents = {}
        for section in agent_sections:
            lines = section.split('\n')
            id = lines[0].strip()
            role = lines[1].split('Role: ')[1] if 'Role: ' in lines[1] else ''
            description = lines[2].split('Description: ')[1] if 'Description: ' in lines[2] else ''
            script_start = next(i for i, line in enumerate(lines) if line.startswith('```python'))
            script_end = next(i for i, line in enumerate(lines[script_start:]) if line.startswith('```') and i > 0) + script_start
            script = '\n'.join(lines[script_start + 1:script_end])
            agents[id] = {"role": role, "description": description, "script": script}
        db.vials.delete_many({})
        db.vials.insert_many([{"id": id, **agent} for id, agent in agents.items()])
        logger.info(f"Imported {len(agents)} vials from {file.filename}")
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to import vials: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/modules", dependencies=[Depends(verify_token)])
async def add_module(file: UploadFile = File(...)):
    try:
        async with aiofiles.open(f"/tmp/{file.filename}", 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        with open(f"/tmp/{file.filename}", 'r') as f:
            text = f.read()
        module_data = {"name": file.filename, "content": text, "timestamp": datetime.datetime.utcnow().isoformat()}
        db.modules.insert_one(module_data)
        logger.info(f"Added module: {file.filename}")
        return {"status": "success", "module": module_data}
    except Exception as e:
        logger.error(f"Failed to add module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    try:
        db.command("ping")
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise Exception(f"Failed to connect to MongoDB: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    mongo_client.close()
    logger.info("Disconnected from MongoDB")
