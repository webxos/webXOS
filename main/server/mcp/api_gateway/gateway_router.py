# main/server/mcp/api_gateway/gateway_router.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
from ..utils.error_handler import handle_api_error
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.rate_limiter import RateLimiter
import httpx
import os

app = FastAPI(title="Vial MCP API Gateway")
metrics = PerformanceMetrics()
rate_limiter = RateLimiter(limit=100, window=60)  # 100 requests per minute
SERVICE_REGISTRY = {
    "auth": os.getenv("AUTH_SERVICE_URL", "http://localhost:8002"),
    "quantum": os.getenv("QUANTUM_SERVICE_URL", "http://localhost:8001"),
    "wallet": os.getenv("WALLET_SERVICE_URL", "http://localhost:8000/wallet"),
    "vials": os.getenv("VIALS_SERVICE_URL", "http://localhost:8000/vials"),
    "ai": os.getenv("AI_SERVICE_URL", "http://localhost:8000/ai")
}

class APIRequest(BaseModel):
    service: str
    endpoint: str
    method: str = "GET"
    data: Dict[str, Any] = {}

@app.post("/route")
async def route_request(request: APIRequest):
    with metrics.track_span("route_request", {"service": request.service, "endpoint": request.endpoint}):
        try:
            if not rate_limiter.allow():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            service_url = SERVICE_REGISTRY.get(request.service)
            if not service_url:
                raise HTTPException(status_code=400, detail="Unknown service")
            url = f"{service_url}/{request.endpoint}"
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    json=request.data if request.method in ["POST", "PUT"] else None,
                    params=request.data if request.method == "GET" else None
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            handle_api_error(e, endpoint=f"{request.service}/{request.endpoint}")
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.get("/health")
async def health_check():
    with metrics.track_span("gateway_health_check"):
        try:
            results = {}
            async with httpx.AsyncClient() as client:
                for service, url in SERVICE_REGISTRY.items():
                    response = await client.get(f"{url}/health")
                    results[service] = response.json()
            return {"status": "healthy", "services": results}
        except Exception as e:
            handle_api_error(e, endpoint="health")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
