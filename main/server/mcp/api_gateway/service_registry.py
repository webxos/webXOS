# main/server/mcp/api_gateway/service_registry.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import os

app = FastAPI(title="Vial MCP Service Registry")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class Service(BaseModel):
    name: str
    url: str
    health_check: str
    metadata: Dict = {}

class ServiceResponse(BaseModel):
    service_id: str
    name: str
    url: str
    health_check: str
    metadata: Dict
    timestamp: str

@app.post("/services", response_model=ServiceResponse)
async def register_service(service: Service, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("register_service", {"name": service.name}):
        try:
            metrics.verify_token(token)
            service_data = service.dict()
            service_data["timestamp"] = datetime.utcnow()
            service_id = db_manager.insert_one("services", service_data)
            return ServiceResponse(service_id=service_id, **service_data)
        except Exception as e:
            handle_generic_error(e, context="register_service")
            raise HTTPException(status_code=500, detail=f"Failed to register service: {str(e)}")

@app.get("/services", response_model=List[ServiceResponse])
async def list_services(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("list_services"):
        try:
            metrics.verify_token(token)
            services = db_manager.find_many("services", {})
            return [ServiceResponse(service_id=str(service["_id"]), **service) for service in services]
        except Exception as e:
            handle_generic_error(e, context="list_services")
            raise HTTPException(status_code=500, detail=f"Failed to list services: {str(e)}")

@app.get("/services/{name}", response_model=ServiceResponse)
async def get_service(name: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_service", {"name": name}):
        try:
            metrics.verify_token(token)
            service = db_manager.find_one("services", {"name": name})
            if not service:
                raise HTTPException(status_code=404, detail="Service not found")
            return ServiceResponse(service_id=str(service["_id"]), **service)
        except Exception as e:
            handle_generic_error(e, context="get_service")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve service: {str(e)}")
