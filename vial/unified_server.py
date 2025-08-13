import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pymongo import MongoClient
from pymongo.errors import ConnectionError
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
import hashlib
import uuid
from dotenv import load_dotenv
import httpx
from fastapi.responses import JSONResponse

# Setup logging
logging.basicConfig(filename='/db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL', 'http://localhost:8001')
VIAL_SERVICE_URL = os.getenv('VIAL_SERVICE_URL', 'http://localhost:8002')
BLOCKCHAIN_SERVICE_URL = os.getenv('BLOCKCHAIN_SERVICE_URL', 'http://localhost:8003')
WALLET_SERVICE_URL = os.getenv('WALLET_SERVICE_URL', 'http://localhost:8004')
QUANTUM_SERVICE_URL = os.getenv('QUANTUM_SERVICE_URL', 'http://localhost:8005')
VIAL_VERSION = '2.8'

app = FastAPI(title="Vial MCP API Gateway", version=VIAL_VERSION)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# MongoDB connection for gateway metadata
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['vial_mcp']
    client.admin.command('ping')
except ConnectionError as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    raise Exception(f"MongoDB connection failed: {str(e)}")

class ErrorLog(BaseModel):
    error: str
    stack: str
    endpoint: str
    timestamp: str
    source: str
    rawResponse: str

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
    address: str | None
    balance: float
    hash: str | None
    webxos: float
    transactions: list

class BlockchainRequest(BaseModel):
    type: str
    data: dict
    timestamp: str
    hash: str

class QuantumLinkRequest(BaseModel):
    vials: list

async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AUTH_SERVICE_URL}/auth/validate/{token}")
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            return response.json()['user_id']
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error at {request.url}: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": f"Server error: {str(exc)}"})

@app.get("/api/health")
async def health_check():
    try:
        client.admin.command('ping')
        async with httpx.AsyncClient() as client:
            services = [
                ('Authentication', f"{AUTH_SERVICE_URL}/auth/health"),
                ('Vial Management', f"{VIAL_SERVICE_URL}/vials/health"),
                ('Blockchain', f"{BLOCKCHAIN_SERVICE_URL}/blockchain/health"),
                ('Wallet', f"{WALLET_SERVICE_URL}/wallet/health"),
                ('Quantum Link', f"{QUANTUM_SERVICE_URL}/quantum/health")
            ]
            service_status = []
            for name, url in services:
                try:
                    response = await client.get(url, timeout=2.0)
                    service_status.append(name if response.status_code == 200 else f"{name} (down)")
                except:
                    service_status.append(f"{name} (down)")
            return {"status": "healthy", "mongo": True, "version": VIAL_VERSION, "services": service_status}
    except ConnectionError as e:
        logger.error(f"Health check failed: MongoDB connection error: {str(e)}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "mongo": False, "version": VIAL_VERSION, "services": []})

@app.post("/api/auth/login")
async def authenticate(auth: AuthRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{AUTH_SERVICE_URL}/auth/login", json=auth.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.post("/api/auth/api-key/generate")
async def generate_api_key(auth: AuthRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{AUTH_SERVICE_URL}/auth/api-key/generate", json=auth.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"API key generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API key generation failed: {str(e)}")

@app.post("/api/log_error")
async def log_error(error: ErrorLog):
    try:
        db['errors'].insert_one(error.dict())
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Error logging failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging failed: {str(e)}")

@app.post("/api/vials/{vialId}/prompt")
async def send_prompt(prompt: PromptRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{VIAL_SERVICE_URL}/vials/{prompt.vialId}/prompt", json=prompt.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Prompt error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prompt processing failed: {str(e)}")

@app.post("/api/vials/{vialId}/task")
async def send_task(task: TaskRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{VIAL_SERVICE_URL}/vials/{task.vialId}/task", json=task.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Task error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task processing failed: {str(e)}")

@app.put("/api/vials/{vialId}/config")
async def set_config(config: ConfigRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(f"{VIAL_SERVICE_URL}/vials/{config.vialId}/config", json=config.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@app.delete("/api/vials/void")
async def void_vials(user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{VIAL_SERVICE_URL}/vials/void")
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Void error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Void failed: {str(e)}")

@app.post("/api/wallet/create")
async def create_wallet(wallet: WalletRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{WALLET_SERVICE_URL}/wallet/create", json=wallet.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Wallet creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet creation failed: {str(e)}")

@app.post("/api/wallet/import")
async def import_wallet(wallet: WalletRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{WALLET_SERVICE_URL}/wallet/import", json=wallet.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Wallet import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet import failed: {str(e)}")

@app.post("/api/wallet/transaction")
async def wallet_transaction(wallet: WalletRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{WALLET_SERVICE_URL}/wallet/transaction", json=wallet.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Wallet transaction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet transaction failed: {str(e)}")

@app.post("/api/blockchain/transaction")
async def add_to_blockchain(block: BlockchainRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BLOCKCHAIN_SERVICE_URL}/blockchain/transaction", json=block.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Blockchain error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blockchain error: {str(e)}")

@app.post("/api/quantum/link")
async def quantum_link(link: QuantumLinkRequest, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{QUANTUM_SERVICE_URL}/quantum/link", json=link.dict())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Quantum link error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum link failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
