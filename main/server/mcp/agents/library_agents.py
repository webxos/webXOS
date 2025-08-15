# main/server/mcp/agents/library_agent.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime

app = FastAPI(title="Vial MCP Library Agent")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class LibraryItem(BaseModel):
    user_id: str
    title: str
    content: str
    tags: List[str] = []
    category: Optional[str] = None

class LibraryItemResponse(LibraryItem):
    item_id: str
    timestamp: str

@app.post("/agents/library", response_model=LibraryItemResponse)
async def add_library_item(item: LibraryItem, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("add_library_item", {"user_id": item.user_id}):
        try:
            metrics.verify_token(token)
            item_dict = item.dict()
            item_dict["timestamp"] = datetime.utcnow()
            item_id = db_manager.insert_one("library", item_dict)
            return LibraryItemResponse(item_id=item_id, **item_dict)
        except Exception as e:
            handle_generic_error(e, context="add_library_item")
            raise HTTPException(status_code=500, detail=f"Failed to add library item: {str(e)}")

@app.get("/agents/library/{user_id}", response_model=List[LibraryItemResponse])
async def get_library_items(user_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_library_items", {"user_id": user_id}):
        try:
            metrics.verify_token(token)
            items = db_manager.find_many("library", {"user_id": user_id})
            return [LibraryItemResponse(item_id=str(item["_id"]), **item) for item in items]
        except Exception as e:
            handle_generic_error(e, context="get_library_items")
            raise HTTPException(status_code=500, detail=f"Failed to fetch library items: {str(e)}")

@app.delete("/agents/library/{item_id}")
async def delete_library_item(item_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("delete_library_item", {"item_id": item_id}):
        try:
            metrics.verify_token(token)
            result = db_manager.delete_one("library", {"_id": item_id})
            if result == 0:
                raise HTTPException(status_code=404, detail="Library item not found")
            return {"status": "success", "message": "Library item deleted"}
        except Exception as e:
            handle_generic_error(e, context="delete_library_item")
            raise HTTPException(status_code=500, detail=f"Failed to delete library item: {str(e)}")
