from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sqlite3
import os
import logging
from datetime import datetime
import hashlib
import uuid
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    filename="/db/errorlog.md",
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebXOS Unified Server", version="2.8", root_path="/api")

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
    logger.info("MongoDB connected successfully")
except ConnectionFailure as e:
    logger.error(f"MongoDB not available, using SQLite fallback: {str(e)}")
    use_mongo = False

# SQLite fallback
SQLITE_DB = "/vial/database.sqlite"
def init_sqlite():
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS wallets
                         (user_id TEXT PRIMARY KEY, address TEXT, balance REAL, hash TEXT, transactions TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS logs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, error TEXT, stack TEXT, source TEXT, endpoint TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, data TEXT, timestamp TEXT, hash TEXT)''')
        conn.commit()
        conn.close()
        logger.info("SQLite database initialized successfully")
    except Exception as e:
        logger.error(f"SQLite initialization failed: {str(e)}")
        raise

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
    endpoint: Optional[str]

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
@app.get("/api/health")
async def health_check():
    try:
        if use_mongo:
            client.admin.command('ping')
        logger.info("Health check successful")
        return {"status": "healthy", "mongo": use_mongo, "version": "2.8"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/auth")
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
        logger.info(f"Authenticated user: {auth.userId}, wallet: {wallet_address}")
        return {
            "apiKey": api_key,
            "walletAddress": wallet_address,
            "walletHash": wallet_hash
        }
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.post("/api/prompt")
async def process_prompt(prompt: PromptRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        logger.error("Invalid or missing API key for /api/prompt")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        response_text = f"Processed prompt for {prompt.vialId}: {prompt.prompt[:50]}..."
        logger.info(f"Processed prompt for vial {prompt.vialId}, blockHash: {prompt.blockHash}")
        return {"response": response_text, "blockHash": prompt.blockHash}
    except Exception as e:
        logger.error(f"Prompt processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prompt processing failed: {str(e)}")

@app.post("/api/quantum_link")
async def quantum_link(link: QuantumLinkRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        logger.error("Invalid or missing API key for /api/quantum_link")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        statuses = ["running" for _ in link.vials]
        latencies = [50 + i * 10 for i in range(len(link.vials))]
        logger.info(f"Quantum link established for vials: {', '.join(link.vials)}")
        return {"statuses": statuses, "latencies": latencies}
    except Exception as e:
        logger.error(f"Quantum link failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum link failed: {str(e)}")

@app.post("/api/wallet")
async def update_wallet(wallet: WalletRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        logger.error("Invalid or missing API key for /api/wallet")
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
        logger.info(f"Wallet updated for user: {wallet.userId}")
        return {"status": "wallet updated"}
    except Exception as e:
        logger.error(f"Wallet update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet update failed: {str(e)}")

@app.post("/api/import_wallet")
async def import_wallet(wallet: ImportWalletRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        logger.error("Invalid or missing API key for /api/import_wallet")
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
        logger.info(f"Wallet imported for user: {wallet.userId}")
        return {"status": "wallet imported"}
    except Exception as e:
        logger.error(f"Wallet import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet import failed: {str(e)}")

@app.post("/api/log_error")
async def log_error(error: ErrorLogRequest):
    try:
        error_data = {
            "timestamp": error.timestamp,
            "error": error.error,
            "stack": error.stack,
            "source": error.source,
            "endpoint": error.endpoint or "unknown"
        }
        if use_mongo:
            logs_collection.insert_one(error_data)
        else:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO logs (timestamp, error, stack, source, endpoint) VALUES (?, ?, ?, ?, ?)",
                (error.timestamp, error.error, error.stack, error.source, error.endpoint or "unknown")
            )
            conn.commit()
            conn.close()
        logger.info(f"Error logged: {error.error}")
        return {"status": "error logged"}
    except Exception as e:
        logger.error(f"Error logging failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging failed: {str(e)}")

@app.post("/api/blockchain")
async def add_to_blockchain(block: BlockchainRequest, api_key: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    if not api_key:
        logger.error("Invalid or missing API key for /api/blockchain")
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
        logger.info(f"Block added to blockchain: {block.hash}")
        return {"status": "block added", "hash": block.hash}
    except Exception as e:
        logger.error(f"Blockchain addition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blockchain addition failed: {str(e)}")

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return {"detail": exc.detail, "status_code": exc.status_code}

if __name__ == "__main__":
    try:
        logger.info("Starting WebXOS Unified Server on 0.0.0.0:8000")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise
