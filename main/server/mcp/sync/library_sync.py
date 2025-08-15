# main/server/mcp/sync/library_sync.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import requests
import os

app = FastAPI(title="Vial MCP Library Sync")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class LibrarySyncRequest(BaseModel):
    user_id: str
    item_id: str
    node_id: str
    item_data: Dict

class LibrarySyncResponse(BaseModel):
    user_id: str
    item_id: str
    node_id: str
    timestamp: str

@app.post("/sync/library", response_model=LibrarySyncResponse)
async def sync_library(request: LibrarySyncRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("sync_library", {"user_id": request.user_id, "item_id": request.item_id}):
        try:
            metrics.verify_token(token)
            existing_item = db_manager.find_one("library", {"_id": request.item_id, "user_id": request.user_id})
            if not existing_item:
                db_manager.insert_one("library", {**request.item_data, "_id": request.item_id, "user_id": request.user_id})
            else:
                db_manager.update_one("library", {"_id": request.item_id}, request.item_data)

            sync_data = {
                "user_id": request.user_id,
                "item_id": request.item_id,
                "node_id": request.node_id,
                "timestamp": datetime.utcnow()
            }
            db_manager.insert_one("library_sync", sync_data)

            # Notify other nodes
            nodes = os.getenv("SYNC_NODES", "").split(",")
            for node in nodes:
                if node and node != request.node_id:
                    try:
                        requests.post(
                            f"{node}/sync/library",
                            json={**sync_data, "item_data": request.item_data},
                            timeout=5
                        )
                    except requests.RequestException:
                        metrics.record_error("sync_library_node_failure", f"Failed to sync with {node}")

            return LibrarySyncResponse(**sync_data)
        except Exception as e:
            handle_generic_error(e, context="sync_library")
            raise HTTPException(status_code=500, detail=f"Failed to sync library item: {str(e)}")
