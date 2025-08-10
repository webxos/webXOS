from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import sqlite3
import uuid
import os
import torch
import logging
from vial_manager import VialManager
from agent1 import Agent1
from agent2 import Agent2
from agent3 import Agent3
from agent4 import Agent4

app = FastAPI()

logging.basicConfig(level=logging.INFO, filename='server.log', format='%(asctime)s %(levelname)s:%(message)s')

class AuthRequest(BaseModel):
    client: str
    deviceId: str
    sessionId: str
    networkId: str

class VoidRequest(BaseModel):
    networkId: str

@app.get("/api/mcp/ping")
async def ping():
    return {"status": "ok"}

@app.get("/api/mcp/health")
async def health():
    return {"status": "ok", "version": "1.7"}

@app.post("/api/mcp/auth")
async def auth(request: AuthRequest):
    try:
        network_id = request.networkId
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        token = str(uuid.uuid4())
        cursor.execute('INSERT OR REPLACE INTO auth (network_id, token) VALUES (?, ?)', (network_id, token))
        cursor.execute('INSERT OR REPLACE INTO wallets (network_id, address, balance) VALUES (?, ?, ?)', (network_id, str(uuid.uuid4()), 0.0))
        conn.commit()
        conn.close()
        return {"token": token, "address": str(uuid.uuid4())}
    except Exception as e:
        logging.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/void")
async def void_request(request: VoidRequest):
    try:
        network_id = request.networkId
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM auth WHERE network_id = ?', (network_id,))
        cursor.execute('DELETE FROM wallets WHERE network_id = ?', (network_id,))
        cursor.execute('DELETE FROM vials WHERE network_id = ?', (network_id,))
        conn.commit()
        conn.close()
        return {"status": "voided"}
    except Exception as e:
        logging.error(f"Void error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/train")
async def train(networkId: str = Form(...), code: str = Form(...), isPython: bool = Form(...)):
    try:
        if not code:
            raise HTTPException(status_code=400, detail="No code provided for training")
        manager = VialManager(networkId)
        balance = 0.0004
        manager.train_vials(code, isPython)
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE wallets SET balance = balance + ? WHERE network_id = ?', (balance, networkId))
        conn.commit()
        conn.close()
        return {
            "vials": manager.get_vials(),
            "balance": balance
        }
    except Exception as e:
        logging.error(f"Train error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/upload")
async def upload(networkId: str = Form(...), file: UploadFile = File(...)):
    try:
        allowed_extensions = {'.py', '.js', '.txt', '.md'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are allowed")
        file_path = f"/uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filePath": file_path}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# [xaiartifact: v1.7]
