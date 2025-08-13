from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv
import logging
import uuid

load_dotenv()

app = FastAPI(title="Vial MCP Controller API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename="db/errorlog.md", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SECRET_KEY = os.getenv("JWT_SECRET", "secret-key")
ALGORITHM = "HS256"

class AuthRequest(BaseModel):
    userId: str

class PromptRequest(BaseModel):
    vialId: str
    prompt: str
    blockHash: str

class TaskRequest(BaseModel):
    vialId: str
    task: str
    blockHash: str

class ConfigRequest(BaseModel):
    vialId: str
    key: str
    value: str
    blockHash: str

class WalletRequest(BaseModel):
    userId: str
    address: str
    balance: float
    hash: str
    webxos: float

class QuantumLinkRequest(BaseModel):
    vials: List[str]

class BlockchainTransaction(BaseModel):
    type: str
    data: Dict
    timestamp: str
    hash: str

def create_jwt_token(user_id: str):
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mongo": True, "version": "2.8", "services": ["auth", "wallet", "vials"]}

@app.post("/auth/login")
async def login(auth: AuthRequest):
    api_key = f"api-{uuid.uuid4()}"
    wallet_address = f"wallet-{uuid.uuid4()}"
    wallet_hash = f"hash-{uuid.uuid4()}"
    logging.info(f"User {auth.userId} logged in, API key: {api_key}")
    return {
        "apiKey": api_key,
        "userId": auth.userId,
        "walletAddress": wallet_address,
        "walletHash": wallet_hash
    }

@app.post("/auth/api-key/generate")
async def generate_api_key(auth: AuthRequest, user_id: str = Depends(get_current_user)):
    if user_id != auth.userId:
        raise HTTPException(status_code=403, detail="Unauthorized")
    api_key = f"api-{uuid.uuid4()}"
    logging.info(f"New API key generated for user {user_id}: {api_key}")
    return {"apiKey": api_key}

@app.post("/vials/{vial_id}/prompt")
async def send_prompt(vial_id: str, prompt: PromptRequest, user_id: str = Depends(get_current_user)):
    if vial_id not in [f"vial{i+1}" for i in range(4)]:
        raise HTTPException(status_code=400, detail="Invalid vial ID")
    logging.info(f"Prompt sent to {vial_id} by {user_id}: {prompt.prompt}")
    return {"response": f"Prompt processed for {vial_id}"}

@app.post("/vials/{vial_id}/task")
async def assign_task(vial_id: str, task: TaskRequest, user_id: str = Depends(get_current_user)):
    if vial_id not in [f"vial{i+1}" for i in range(4)]:
        raise HTTPException(status_code=400, detail="Invalid vial ID")
    logging.info(f"Task assigned to {vial_id} by {user_id}: {task.task}")
    return {"status": f"Task assigned to {vial_id}"}

@app.put("/vials/{vial_id}/config")
async def set_config(vial_id: str, config: ConfigRequest, user_id: str = Depends(get_current_user)):
    if vial_id not in [f"vial{i+1}" for i in range(4)]:
        raise HTTPException(status_code=400, detail="Invalid vial ID")
    logging.info(f"Config updated for {vial_id} by {user_id}: {config.key}={config.value}")
    return {"status": f"Config updated for {vial_id}"}

@app.delete("/vials/void")
async def void_vials(user_id: str = Depends(get_current_user)):
    logging.info(f"All vials reset by {user_id}")
    return {"status": "All vials reset"}

@app.post("/wallet/create")
async def create_wallet(wallet: WalletRequest, user_id: str = Depends(get_current_user)):
    if user_id != wallet.userId:
        raise HTTPException(status_code=403, detail="Unauthorized")
    logging.info(f"Wallet created for {user_id}: {wallet.address}")
    return {"status": "Wallet created", "address": wallet.address, "webxos": wallet.webxos}

@app.post("/wallet/import")
async def import_wallet(wallet: WalletRequest, user_id: str = Depends(get_current_user)):
    if user_id != wallet.userId:
        raise HTTPException(status_code=403, detail="Unauthorized")
    logging.info(f"Wallet imported for {user_id}: {wallet.address}")
    return {"status": "Wallet imported"}

@app.post("/wallet/transaction")
async def record_transaction(wallet: WalletRequest, user_id: str = Depends(get_current_user)):
    if user_id != wallet.userId:
        raise HTTPException(status_code=403, detail="Unauthorized")
    logging.info(f"Transaction recorded for {user_id}: {wallet.address}")
    return {"status": "Transaction recorded"}

@app.post("/quantum/link")
async def quantum_link(link: QuantumLinkRequest, user_id: str = Depends(get_current_user)):
    if not all(v in [f"vial{i+1}" for i in range(4)] for v in link.vials):
        raise HTTPException(status_code=400, detail="Invalid vial IDs")
    logging.info(f"Quantum link established by {user_id} for vials: {link.vials}")
    return {"statuses": ["running"] * len(link.vials), "latencies": [50, 60, 70, 80][:len(link.vials)]}

@app.post("/blockchain/transaction")
async def blockchain_transaction(transaction: BlockchainTransaction, user_id: str = Depends(get_current_user)):
    logging.info(f"Blockchain transaction recorded by {user_id}: {transaction.type}")
    return {"status": "Transaction recorded"}

@app.post("/log_error")
async def log_error(error_data: Dict, user_id: str = Depends(get_current_user)):
    logging.error(f"Error from {user_id}: {error_data['error']} at {error_data['endpoint']}")
    return {"status": "Error logged"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
