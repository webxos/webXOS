from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
import uuid
import os
from typing import List, Dict

app = FastAPI()
security = HTTPBearer()

# SQLite setup
def init_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vials
                     (id TEXT PRIMARY KEY, status TEXT, code TEXT, code_length INTEGER, is_python BOOLEAN)''')
        c.execute('''CREATE TABLE IF NOT EXISTS wallets
                     (address TEXT PRIMARY KEY, balance REAL, wallet_key TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS auth
                     (key TEXT PRIMARY KEY, token TEXT, timestamp INTEGER)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] SQLITE ERRORS: SQLite Init Error: {str(e)}\nAnalysis: Check database.db permissions\nTraceback: No stack\n")
        raise

init_db()

class AuthData(BaseModel):
    client: str
    deviceId: str
    sessionId: str

class TrainData(BaseModel):
    code: str
    isPython: bool

class WalletData(BaseModel):
    walletKey: str
    balance: float

@app.post("/mcp/auth")
async def auth(data: AuthData):
    try:
        token = str(uuid.uuid4())
        address = str(uuid.uuid4())
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO auth (key, token, timestamp) VALUES (?, ?, ?)",
                  ('master', token, int(new Date().getTime() / 1000)))
        c.execute("INSERT OR REPLACE INTO wallets (address, balance, wallet_key) VALUES (?, ?, ?)",
                  (address, 0.0, str(uuid.uuid4())))
        conn.commit()
        conn.close()
        return {"token": token, "address": address}
    except sqlite3.Error as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] SQLITE ERRORS: SQLite Auth Error: {str(e)}\nAnalysis: Check database.db\nTraceback: No stack\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/void", dependencies=[Depends(security)])
async def void(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DELETE FROM vials")
        c.execute("DELETE FROM auth")
        c.execute("DELETE FROM wallets")
        conn.commit()
        conn.close()
        return {"status": "voided"}
    except sqlite3.Error as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] SQLITE ERRORS: SQLite Void Error: {str(e)}\nAnalysis: Check database.db\nTraceback: No stack\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/train", dependencies=[Depends(security)])
async def train(data: TrainData, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        vials = [
            {"id": f"vial{i+1}", "status": "running", "code": data.code, "code_length": len(data.code), "is_python": data.isPython}
            for i in range(4)
        ]
        for vial in vials:
            c.execute("INSERT OR REPLACE INTO vials (id, status, code, code_length, is_python) VALUES (?, ?, ?, ?, ?)",
                      (vial["id"], vial["status"], vial["code"], vial["code_length"], vial["is_python"]))
        c.execute("UPDATE wallets SET balance = balance + ? WHERE wallet_key = ?",
                  (0.0, credentials.credentials)) # Balance updated client-side
        conn.commit()
        conn.close()
        return {"vials": vials, "balance": 0.0}
    except sqlite3.Error as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] SQLITE ERRORS: SQLite Train Error: {str(e)}\nAnalysis: Check database.db\nTraceback: No stack\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/upload", dependencies=[Depends(security)])
async def upload(file: bytes = File(...), credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"filePath": file_path}
    except Exception as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] POSSIBLE OTHER ERRORS: Upload Error: {str(e)}\nAnalysis: Check uploads directory\nTraceback: No stack\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/validate_wallet", dependencies=[Depends(security)])
async def validate_wallet(data: WalletData, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT balance FROM wallets WHERE wallet_key = ?", (data.walletKey,))
        result = c.fetchone()
        if result and result[0] == data.balance:
            c.execute("UPDATE wallets SET balance = balance + ? WHERE wallet_key = ?",
                      (data.balance, credentials.credentials))
            conn.commit()
            conn.close()
            return {"validatedBalance": data.balance}
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid wallet key or balance")
    except sqlite3.Error as e:
        with open('errorlog.md', 'a') as f:
            f.write(f"[{new Date().toISOString()}] SQLITE ERRORS: SQLite Wallet Validation Error: {str(e)}\nAnalysis: Check database.db\nTraceback: No stack\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/ping")
async def ping():
    return {"status": "ok"}

# xAI Artifact Tags: #VialMCP #WebXOS #FastAPI #SQLite
