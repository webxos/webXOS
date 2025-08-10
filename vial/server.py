from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import sqlite3
import uuid
import os
import torch
import logging
from vial_manager import VialManager
from auth_manager import AuthManager
from export_manager import ExportManager
from langchain_agent import LangChainAgent
from vector_container import VectorContainer

app = FastAPI()

logging.basicConfig(level=logging.INFO, filename='server.log', format='%(asctime)s %(levelname)s:%(message)s')

# Initialize database
def init_db():
    conn = sqlite3.connect('vial.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS auth (network_id TEXT PRIMARY KEY, token TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS wallets (network_id TEXT PRIMARY KEY, address TEXT, balance REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS vials (network_id TEXT PRIMARY KEY, vial_data TEXT)''')
    conn.commit()
    conn.close()

init_db()

class AuthRequest(BaseModel):
    client: str
    deviceId: str
    sessionId: str
    networkId: str

class VoidRequest(BaseModel):
    networkId: str

class GitRequest(BaseModel):
    command: str

class CommsRequest(BaseModel):
    message: str

@app.get("/api/mcp/ping")
async def ping():
    return {"status": "ok"}

@app.get("/api/mcp/health")
async def health():
    return {"status": "ok", "version": "1.7"}

@app.post("/api/mcp/auth")
async def auth(request: AuthRequest):
    try:
        auth_manager = AuthManager()
        token = auth_manager.authenticate(request.client, request.deviceId, request.sessionId, request.networkId)
        if not token:
            raise HTTPException(status_code=401, detail="Authentication failed")
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO auth (network_id, token) VALUES (?, ?)', (request.networkId, token))
        cursor.execute('INSERT OR REPLACE INTO wallets (network_id, address, balance) VALUES (?, ?, ?)', 
                      (request.networkId, auth_manager.get_wallet_address(), 0.0))
        conn.commit()
        conn.close()
        return {"token": token, "address": auth_manager.get_wallet_address()}
    except Exception as e:
        logging.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/void")
async def void_request(request: VoidRequest):
    try:
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM auth WHERE network_id = ?', (request.networkId,))
        cursor.execute('DELETE FROM wallets WHERE network_id = ?', (request.networkId,))
        cursor.execute('DELETE FROM vials WHERE network_id = ?', (request.networkId,))
        conn.commit()
        conn.close()
        return {"status": "voided"}
    except Exception as e:
        logging.error(f"Void error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/train")
async def train(networkId: str = Form(...), file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only .md files are allowed")
        code = await file.read()
        code = code.decode('utf-8')
        if "## WEBXOS Tokenization Tag:" not in code:
            raise HTTPException(status_code=400, detail="Invalid .md: Missing WEBXOS Tokenization Tag")
        vector_container = VectorContainer()
        chunks = vector_container.chunk_md(code)
        metadata = {"network_id": networkId, "timestamp": str(time.time())}
        chunked_code = '\n'.join(vector_container.add_metadata(chunk, metadata) for chunk in chunks)
        manager = VialManager(networkId)
        balance = 0.0004
        manager.train_vials(chunked_code, isPython=False)
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
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only .md files are allowed")
        content = await file.read()
        content_str = content.decode('utf-8')
        if "## WEBXOS Tokenization Tag:" not in content_str:
            raise HTTPException(status_code=400, detail="Invalid .md: Missing WEBXOS Tokenization Tag")
        vector_container = VectorContainer()
        chunks = vector_container.chunk_md(content_str)
        metadata = {"network_id": networkId, "timestamp": str(time.time())}
        chunked_content = '\n'.join(vector_container.add_metadata(chunk, metadata) for chunk in chunks)
        file_path = f"/uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(chunked_content)
        return {"filePath": file_path}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/mcp/export")
async def export(networkId: str):
    try:
        manager = VialManager(networkId)
        export_manager = ExportManager()
        md_content = export_manager.export_to_md(networkId, manager.get_vials())
        vector_container = VectorContainer()
        chunks = vector_container.chunk_md(md_content)
        metadata = {"network_id": networkId, "timestamp": str(time.time())}
        chunked_md = '\n'.join(vector_container.add_metadata(chunk, metadata) for chunk in chunks)
        vectors_md = "## Vectors\n" + '\n'.join(f"- Chunk {i}: {vector_container.vectorize(chunk)}" for i, chunk in enumerate(chunks))
        final_md = chunked_md + '\n' + vectors_md
        return {"markdown": final_md}
    except Exception as e:
        logging.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/git")
async def git_command(request: GitRequest):
    try:
        agent = LangChainAgent()
        result = agent.handle_git(request.command)
        return {"result": result}
    except Exception as e:
        logging.error(f"Git error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/mcp/comms")
async def comms(request: CommsRequest):
    try:
        agent = LangChainAgent()
        response = agent.handle_message(request.message)
        return {"response": response}
    except Exception as e:
        logging.error(f"Comms error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# [xaiartifact: v1.7]
