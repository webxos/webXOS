# main/server/mcp/sync/auth_sync.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import requests
import os

app = FastAPI(title="Vial MCP Auth Sync")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class AuthSyncRequest(BaseModel):
    user_id: str
    token: str
    node_id: str

class AuthSyncResponse(BaseModel):
    user_id: str
    token: str
    node_id: str
    timestamp: str

@app.post("/sync/auth", response_model=AuthSyncResponse)
async def sync_auth(request: AuthSyncRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("sync_auth", {"user_id": request.user_id, "node_id": request.node_id}):
        try:
            metrics.verify_token(token)
            existing_session = db_manager.find_one("sessions", {"user_id": request.user_id, "token": request.token})
            if not existing_session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            sync_data = {
                "user_id": request.user_id,
                "token": request.token,
                "node_id": request.node_id,
                "timestamp": datetime.utcnow()
            }
            db_manager.insert_one("auth_sync", sync_data)
            
            # Notify other nodes
            nodes = os.getenv("SYNC_NODES", "").split(",")
            for node in nodes:
                if node and node != request.node_id:
                    try:
                        requests.post(
                            f"{node}/sync/auth",
                            json=sync_data,
                            timeout=5
                        )
                    except requests.RequestException:
                        metrics.record_error("sync_auth_node_failure", f"Failed to sync with {node}")

            return AuthSyncResponse(**sync_data)
        except Exception as e:
            handle_generic_error(e, context="sync_auth")
            raise HTTPException(status_code=500, detail=f"Failed to sync auth: {str(e)}")
