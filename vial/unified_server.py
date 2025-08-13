from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sqlite3
import os
from datetime import datetime
import hashlib
import uuid
from typing import List, Dict, Optional

app = FastAPI(title="WebXOS Unified Server", version="2.8")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["webxos"]
    wallets_collection = db["wallets"]
    logs_collection = db["logs"]
    blockchain_collection = db["blockchain"]
    use_mongo = True
except ConnectionFailure:
    print("MongoDB not available, using SQLite fallback")
    use_mongo = False

# SQLite fallback
SQLITE_DB = "database.sqlite"
def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS wallets
                     (user_id TEXT PRIMARY KEY, address TEXT, balance REAL, hash TEXT, transactions TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, error TEXT, stack TEXT, source TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, data TEXT, timestamp TEXT, hash TEXT)''')
    conn.commit()
    conn.close()

if not use_mongo:
    init_sqlite()

# Pydantic models
class AuthRequest(BaseModel):
    userId: str

class PromptRequest(BaseModel):
    vialId: str
    prompt: str
    blockHash: str

class QuantumLinkRequest(BaseModel):
    vials: List[str]

class WalletRequest(BaseModel):
    userId: str
    address: Optional[str]
    balance: float
    hash: Optional[str]
    webxos: float
    transactions: List[Dict]

class ImportWalletRequest(BaseModel):
    userId: str
    address: str
    balance: float
    hash: str
    webxos: float
    transactions: List[Dict]

class ErrorLogRequest(BaseModel):
    error: str
    stack: str
    timestamp: str
    source: str

class BlockchainRequest(BaseModel):
    type: str
    data: Dict
    timestamp: str
    hash: str

# Utility functions
def generate_wallet_address():
    return f"webxos_{uuid.uuid4().hex[:32]}"

def generate_wallet_hash():
    return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

# Endpoints
@app.get("/health")
async def health_check():
    try:
        if use_mongo:
            client.admin.command('ping')
        return {"status": "healthy", "mongo": use_mongo, "version": "2.8"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/auth")
async def authenticate(auth: AuthRequest):
    try:
        wallet_address = generate_wallet_address()
        wallet_hash = generate_wallet_hash()
        api_key = f"api_{uuid.uuid4().hex}"
        wallet_data = {
            "user_id": auth.userId,
            "address": wallet_address,
            "balance": 0.0,
            "hash": wallet_hash,
            "webxos": 0.0000,
            "transactions": []
        }
        if use_mongo:
            wallets_collection.update_one(
                {"user_id": auth.userId},
                {"$set": wallet_data},
                upsert=True
            )
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO wallets (user_id, address, balance, hash, transactions) VALUES (?, ?, ?, ?, ?)",
                (auth.userId, wallet_address, 0.0, wallet_hash, "[]")
            )
            conn.commit()
            conn.close()
        return {
            "apiKey": api_key,
            "walletAddress": wallet_address,
            "walletHash": wallet_hash
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.post("/prompt")
async def process_prompt(prompt: PromptRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        # Simulate chatbot response
        response_text = f"Processed prompt for {prompt.vialId}: {prompt.prompt[:50]}..."
        return {"response": response_text, "blockHash": prompt.blockHash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt processing failed: {str(e)}")

@app.post("/quantum_link")
async def quantum_link(link: QuantumLinkRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        statuses = ["running" for _ in link.vials]
        latencies = [50 + i * 10 for i in range(len(link.vials))]
        return {"statuses": statuses, "latencies": latencies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum link failed: {str(e)}")

@app.post("/wallet")
async def update_wallet(wallet: WalletRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        wallet_data = {
            "user_id": wallet.userId,
            "address": wallet.address,
            "balance": wallet.balance,
            "hash": wallet.hash,
            "webxos": wallet.webxos,
            "transactions": wallet.transactions
        }
        if use_mongo:
            wallets_collection.update_one(
                {"user_id": wallet.userId},
                {"$set": wallet_data},
                upsert=True
            )
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO wallets (user_id, address, balance, hash, transactions) VALUES (?, ?, ?, ?, ?)",
                (wallet.userId, wallet.address, wallet.balance, wallet.hash, str(wallet.transactions))
            )
            conn.commit()
            conn.close()
        return {"status": "wallet updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wallet update failed: {str(e)}")

@app.post("/import_wallet")
async def import_wallet(wallet: ImportWalletRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        wallet_data = {
            "user_id": wallet.userId,
            "address": wallet.address,
            "balance": wallet.balance,
            "hash": wallet.hash,
            "webxos": wallet.webxos,
            "transactions": wallet.transactions
        }
        if use_mongo:
            wallets_collection.update_one(
                {"user_id": wallet.userId},
                {"$set": wallet_data},
                upsert=True
            )
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO wallets (user_id, address, balance, hash, transactions) VALUES (?, ?, ?, ?, ?)",
                (wallet.userId, wallet.address, wallet.balance, wallet.hash, str(wallet.transactions))
            )
            conn.commit()
            conn.close()
        return {"status": "wallet imported"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wallet import failed: {str(e)}")

@app.post("/log_error")
async def log_error(error: ErrorLogRequest):
    try:
        error_data = {
            "timestamp": error.timestamp,
            "error": error.error,
            "stack": error.stack,
            "source": error.source
        }
        if use_mongo:
            logs_collection.insert_one(error_data)
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO logs (timestamp, error, stack, source) VALUES (?, ?, ?, ?)",
                (error.timestamp, error.error, error.stack, error.source)
            )
            conn.commit()
            conn.close()
        return {"status": "error logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging failed: {str(e)}")

@app.post("/blockchain")
async def add_to_blockchain(block: BlockchainRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        block_data = {
            "type": block.type,
            "data": block.data,
            "timestamp": block.timestamp,
            "hash": block.hash
        }
        if use_mongo:
            blockchain_collection.insert_one(block_data)
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO blockchain (type, data, timestamp, hash) VALUES (?, ?, ?, ?)",
                (block.type, str(block.data), block.timestamp, block.hash)
            )
            conn.commit()
            conn.close()
        return {"status": "block added", "hash": block.hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain addition failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
