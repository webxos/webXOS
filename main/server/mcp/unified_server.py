# main/server/mcp/unified_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api_gateway.service_registry import ServiceRegistry
from .auth.mcp_server_auth import authenticate_user
from .utils.webxos_balance import WebXOSBalance
from .utils.health_check import HealthCheck
from .utils.error_handler import handle_error

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

registry = ServiceRegistry()
balance = WebXOSBalance()
health = HealthCheck()

@app.post("/mcp/auth")
async def auth_endpoint(data: dict):
    try:
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            raise ValueError("Username and password are required")
        token, vials = await authenticate_user(username, password)
        return {"access_token": token, "vials": vials}
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/status")
async def status_endpoint(data: dict):
    try:
        return await health.get_health()
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/checklist")
async def checklist_endpoint(data: dict):
    try:
        return await health.check_system()
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/api_key")
async def api_key_endpoint(data: dict):
    try:
        return {"result": "mock_api_key_123"}
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/import")
async def import_endpoint(data: dict):
    try:
        return {"result": {"status": "imported"}}
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/export")
async def export_endpoint(data: dict):
    try:
        return {"result": "# Vial Data\nExported at 03:07 AM EDT, August 15, 2025"}
    except Exception as e:
        return handle_error(e)
