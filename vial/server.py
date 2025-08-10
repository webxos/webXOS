from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict
import jwt
import sqlite3
import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from vial.models import Vial, Wallet
from vial.middleware import verify_token
from vial.tools.vial_manager import VialManager

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthRequest(BaseModel):
    client: str
    deviceId: str
    sessionId: str
    networkId: str

class TrainRequest(BaseModel):
    code: str
    isPython: bool
    networkId: str

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

vial_manager = VialManager()

@app.post("/mcp/auth")
async def auth(auth_request: AuthRequest):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        token = jwt.encode({
            'client': auth_request.client,
            'deviceId': auth_request.deviceId,
            'sessionId': auth_request.sessionId,
            'networkId': auth_request.networkId,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }, 'secret', algorithm='HS256')
        address = f"0x{auth_request.deviceId[:10]}"
        cursor.execute("INSERT OR REPLACE INTO wallets (address, balance) VALUES (?, ?)", (address, 0))
        conn.commit()
        vials = [
            Vial(id=f"vial{i+1}", status="stopped", code="import torch\nimport torch.nn as nn\n\nclass VialAgent(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    def forward(self, x):\n        return torch.sigmoid(self.fc(x))\n\nmodel = VialAgent()", codeLength=0, isPython=True, webxosHash=f"{auth_request.sessionId}-{i+1}", wallet=Wallet(address=f"0x{vial_manager.generate_uuid()[:10]}", balance=0))
            for i in range(4)
        ]
        for vial in vials:
            cursor.execute("INSERT OR REPLACE INTO vials (id, status, code, codeLength, isPython, webxosHash, walletAddress, walletBalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (vial.id, vial.status, vial.code, vial.codeLength, vial.isPython, vial.webxosHash, vial.wallet.address, vial.wallet.balance))
        conn.commit()
        conn.close()
        return {"token": token, "address": address}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/mcp/ping")
async def ping():
    return {"status": "ok"}

@app.post("/mcp/void", dependencies=[Depends(verify_token)])
async def void(network_id: str):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vials WHERE webxosHash LIKE ?", (f"{network_id}%",))
        cursor.execute("DELETE FROM wallets")
        conn.commit()
        conn.close()
        return {"status": "voided"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/train", dependencies=[Depends(verify_token)])
async def train(code: str = File(...), isPython: bool = File(...), networkId: str = File(...)):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vials WHERE webxosHash LIKE ?", (f"{networkId}%",))
        vials = [Vial(id=row[0], status=row[1], code=row[2], codeLength=row[3], isPython=row[4], webxosHash=row[5], wallet=Wallet(address=row[6], balance=row[7])) for row in cursor.fetchall()]
        cursor.execute("SELECT balance FROM wallets WHERE address = (SELECT address FROM auth WHERE token = ?)", (request.headers.get("Authorization").split()[1],))
        balance = cursor.fetchone()[0]
        for vial in vials:
            if vial.status == "stopped":
                vial.status = "running"
                vial.code = code
                vial.codeLength = len(code)
                vial.isPython = isPython
                vial.wallet.balance = balance / 4
                cursor.execute("UPDATE vials SET status = ?, code = ?, codeLength = ?, isPython = ?, walletBalance = ? WHERE id = ?",
                               (vial.status, vial.code, vial.codeLength, vial.isPython, vial.wallet.balance, vial.id))
        cursor.execute("UPDATE wallets SET balance = balance + ? WHERE address = (SELECT address FROM auth WHERE token = ?)",
                       (0.1, request.headers.get("Authorization").split()[1]))
        conn.commit()
        conn.close()
        return {"vials": [vial.dict() for vial in vials], "balance": balance + 0.1}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/upload", dependencies=[Depends(verify_token)])
async def upload(file: UploadFile = File(...), networkId: str = File(...)):
    try:
        content = await file.read()
        if len(content) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 1MB")
        file_path = f"/uploads/{networkId}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        return {"filePath": file_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/pytorch", dependencies=[Depends(verify_token)])
async def pytorch_train(networkId: str, input_data: List[float]):
    try:
        model = PyTorchModel()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return {"output": output.tolist(), "networkId": networkId}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
