from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.searchbot

class AuthRequest(BaseModel):
    userId: str

class Transaction(BaseModel):
    type: str
    query: str | None = None
    vialId: str | None = None
    file: str | None = None
    apiKey: str | None = None
    timestamp: str

class ErrorLog(BaseModel):
    error: str
    timestamp: str

class QueryLog(BaseModel):
    query: str
    timestamp: str

async def create_user_profile(apiKey: str, userId: str):
    profile = {
        "apiKey": apiKey,
        "userId": userId,
        "created_at": datetime.utcnow(),
        "balance": 0.0,
        "reputation": 0,
        "query_count": 0
    }
    await db.users.update_one({"apiKey": apiKey}, {"$setOnInsert": profile}, upsert=True)

@app.post("/auth")
async def authenticate(auth: AuthRequest):
    apiKey = hashlib.sha256(auth.userId.encode()).hexdigest()
    await create_user_profile(apiKey, auth.userId)
    await db.security_logs.insert_one({
        "apiKey": apiKey,
        "event": "authentication_success",
        "timestamp": datetime.utcnow()
    })
    return {"apiKey": apiKey}

@app.get("/vials")
async def get_vials(request: Request):
    apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not apiKey or not await db.users.find_one({"apiKey": apiKey}):
        raise HTTPException(status_code=401, detail="Unauthorized")
    with open("nano_gpt_bots.md", "r") as f:
        return f.read()

@app.post("/import")
async def import_vials(request: Request):
    apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not apiKey or not await db.users.find_one({"apiKey": apiKey}):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    form = await request.form()
    file = form.get("file")
    if not file or not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    content = await file.read()
    text = content.decode()
    agentSections = text.split("## Agent ")[1:]
    if len(agentSections) != 4:
        raise HTTPException(status_code=400, detail=f"Expected 4 vials, found {len(agentSections)}")
    
    agents = {}
    for section in agentSections:
        lines = section.split('\n')
        id = lines[0].strip()
        role = lines[1].split("Role: ")[1].strip() if "Role: " in lines[1] else ""
        description = lines[2].split("Description: ")[1].strip() if "Description: " in lines[2] else ""
        script_start = next((i for i, line in enumerate(lines) if line.startswith("```python")), -1)
        script = ""
        if script_start != -1:
            script_end = next((i for i, line in enumerate(lines[script_start:]) if line.startswith("```") and i > 0), len(lines))
            script = "\n".join(lines[script_start + 1:script_start + script_end])
        agents[id] = {"role": role, "description": description, "script": script}
    
    await db.users.update_one({"apiKey": apiKey}, {"$inc": {"query_count": 1}})
    await db.security_logs.insert_one({
        "apiKey": apiKey,
        "event": "import_vials",
        "file": file.filename,
        "timestamp": datetime.utcnow()
    })
    return {"agents": agents}

@app.post("/wallet")
async def update_wallet(transaction: Transaction, request: Request):
    apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not apiKey or not await db.users.find_one({"apiKey": apiKey}):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    await db.wallet_logs.insert_one({
        "apiKey": apiKey,
        **transaction.dict(exclude_unset=True),
        "timestamp": datetime.utcnow()
    })
    await db.users.update_one({"apiKey": apiKey}, {"$inc": {"query_count": 1}})
    return {"status": "success"}

@app.post("/log_error")
async def log_error(error: ErrorLog, request: Request):
    apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
    if apiKey and await db.users.find_one({"apiKey": apiKey}):
        await db.security_logs.insert_one({
            "apiKey": apiKey,
            "event": "error",
            "error": error.error,
            "timestamp": datetime.utcnow()
        })
    return {"status": "success"}

@app.post("/log_query")
async def log_query(query: QueryLog, request: Request):
    apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not apiKey or not await db.users.find_one({"apiKey": apiKey}):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    await db.history.insert_one({
        "apiKey": apiKey,
        "query": query.query,
        "timestamp": datetime.utcnow()
    })
    await db.users.update_one({"apiKey": apiKey}, {"$inc": {"query_count": 1}})
    return {"status": "success"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
