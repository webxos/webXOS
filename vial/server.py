import asyncio
import os
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet
from auth_manager import AuthManager
from export_manager import ExportManager
from langchain_agent import create_langchain_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vial MCP API", version="2.1")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
wallet = WebXOSWallet()
auth_manager = AuthManager()
vial_manager = VialManager(wallet)
export_manager = ExportManager(vial_manager, wallet)
langchain_agent = create_langchain_agent()

# Authentication
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key and api_key.startswith("Bearer "):
        token = api_key.replace("Bearer ", "")
        if auth_manager.validate_token(token):
            return token
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Pydantic models
class AuthRequest(BaseModel):
    client: str
    deviceId: str
    sessionId: str
    networkId: str

class CommsRequest(BaseModel):
    message: str
    network_id: str

# API Endpoints
@app.post("/auth")
async def authenticate(auth: AuthRequest):
    try:
        token, address = auth_manager.authenticate(auth.networkId, auth.sessionId)
        logger.info(f"Authenticated session: {token}")
        return {"token": token, "address": address}
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Auth error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/void")
async def void_network(token: str = Depends(get_api_key)):
    try:
        auth_manager.void_session(token)
        vial_manager.reset_vials()
        logger.info("Network voided")
        return {"status": "voided"}
    except Exception as e:
        logger.error(f"Void error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Void error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Health check error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_vials(file: UploadFile = File(...), networkId: str = Form(...), token: str = Depends(get_api_key)):
    try:
        if not auth_manager.validate_session(token, networkId):
            raise HTTPException(status_code=403, detail="Invalid network ID")
        
        content = await file.read()
        content_str = content.decode('utf-8')
        
        balance_earned = vial_manager.train_vials(networkId, content_str, file.filename)
        logger.info(f"Trained vials for network: {networkId}")
        return {"vials": vial_manager.get_vials(), "balance": balance_earned}
    except Exception as e:
        logger.error(f"Train error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Train error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
async def export_vials(networkId: str, token: str = Depends(get_api_key)):
    try:
        if not auth_manager.validate_session(token, networkId):
            raise HTTPException(status_code=403, detail="Invalid network ID")
        
        markdown = export_manager.export_to_markdown(token, networkId)
        logger.info(f"Exported vials for network: {networkId}")
        return {"markdown": markdown}
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Export error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), networkId: str = Form(...), token: str = Depends(get_api_key)):
    try:
        if not auth_manager.validate_session(token, networkId):
            raise HTTPException(status_code=403, detail="Invalid network ID")
        
        content = await file.read()
        file_path = f"/uploads/{file.filename}"
        logger.info(f"File uploaded: {file_path}")
        return {"filePath": file_path}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Upload error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/comms_hub")
async def comms_hub(request: CommsRequest, token: str = Depends(get_api_key)):
    try:
        if not auth_manager.validate_session(token, request.network_id):
            raise HTTPException(status_code=403, detail="Invalid network ID")
        
        response = await langchain_agent.arun(request.message)
        logger.info(f"Comms processed: {request.message}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Comms error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** Comms error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
