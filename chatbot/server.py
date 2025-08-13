from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from pydantic import BaseModel
import logging
import re
import hashlib
import datetime
import aiofiles
import os
import sys

# Add /chatbot/db to sys.path to import nlp_model
sys.path.append(os.path.join(os.path.dirname(__file__), 'db'))
from nlp_model import enhance_query

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

@app.post("/api/log_query", dependencies=[Depends(verify_token)])
async def log_query(query_request: QueryRequest, user: dict = Depends(verify_token)):
    try:
        db.queries.insert_one({
            "query": query_request.query,
            "timestamp": query_request.timestamp,
            "userId": user["userId"]
        })
        logger.info(f"Logged query: {query_request.query} for user: {user['userId']}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to log query: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to log query: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wallet", dependencies=[Depends(verify_token)])
async def update_wallet(wallet_request: WalletRequest, user: dict = Depends(verify_token)):
    try:
        db.wallet.update_one(
            {"userId": user["userId"]},
            {"$push": {"transactions": wallet_request.transaction}, "$inc": {"webxos": 0.0001}},
            upsert=True
        )
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

@app.post("/api/enhance_query", dependencies=[Depends(verify_token)])
async def enhance_query(request: EnhanceQueryRequest, user: dict = Depends(verify_token)):
    try:
        enhanced_query = enhance_query(request.query)
        db.queries.insert_one({
            "query": request.query,
            "enhanced_query": enhanced_query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": user["userId"]
        })
        logger.info(f"Enhanced query: {request.query} -> {enhanced_query}")
        return {"enhanced_query": enhanced_query}
    except Exception as e:
        logger.error(f"Failed to enhance query: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to enhance query: {str(e)}",
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
        agent_sections = text.split('## Agent ')[1:]
        if len(agent_sections) != 4:
            logger.error(f"Invalid number of vials: expected 4, found {len(agent_sections)}")
            db.errors.insert_one({
                "error": f"Invalid number of vials: expected 4, found {len(agent_sections)}",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "backend"
            })
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
        logger.info(f"Imported {len(agents)} vials from {file.filename} for user: {user['userId']}")
        db.queries.insert_one({
            "query": f"Imported {file.filename}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": user["userId"]
        })
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to import vials: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to import vials: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/modules", dependencies=[Depends(verify_token)])
async def add_module(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    try:
        async with aiofiles.open(f"/tmp/{file.filename}", 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        with open(f"/tmp/{file.filename}", 'r') as f:
            text = f.read()
        module_data = {
            "name": file.filename,
            "content": text,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": user["userId"]
        }
        db.modules.insert_one(module_data)
        logger.info(f"Added module: {file.filename} for user: {user['userId']}")
        db.queries.insert_one({
            "query": f"Added module {file.filename}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "userId": user["userId"]
        })
        return {"status": "success", "module": module_data}
    except Exception as e:
        logger.error(f"Failed to add module: {str(e)}")
        db.errors.insert_one({
            "error": f"Failed to add module: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "backend"
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    try:
        db.command("ping")
        logger.info("Connected to MongoDB")
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
