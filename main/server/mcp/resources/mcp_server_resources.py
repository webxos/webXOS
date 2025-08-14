# main/server/mcp/resources/mcp_server_resources.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pymongo import MongoClient
import os
import psutil
from datetime import datetime
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(title="Vial MCP Resources Server")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["vial_mcp"]
resources_collection = db["resources"]
metrics = PerformanceMetrics()

class Resource(BaseModel):
    type: str
    usage: float
    total: float
    unit: str

@app.get("/resources", response_model=list[Resource])
async def get_resources(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_resources"):
        try:
            metrics.verify_token(token)
            resources = [
                Resource(type="CPU", usage=psutil.cpu_percent(), total=100.0, unit="%"),
                Resource(type="Memory", usage=psutil.virtual_memory().percent, total=100.0, unit="%"),
                Resource(type="Disk", usage=psutil.disk_usage('/').percent, total=100.0, unit="%")
            ]
            resources_collection.insert_one({
                "timestamp": datetime.utcnow(),
                "resources": [r.dict() for r in resources]
            })
            return resources
        except Exception as e:
            handle_generic_error(e, context="get_resources")
            raise HTTPException(status_code=500, detail=f"Failed to fetch resources: {str(e)}")

@app.get("/resources/history")
async def get_resources_history(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_resources_history"):
        try:
            metrics.verify_token(token)
            history = list(resources_collection.find().sort("timestamp", -1).limit(10))
            return [{"timestamp": h["timestamp"], "resources": h["resources"]} for h in history]
        except Exception as e:
            handle_generic_error(e, context="get_resources_history")
            raise HTTPException(status_code=500, detail=f"Failed to fetch resources history: {str(e)}")
