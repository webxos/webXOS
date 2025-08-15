# main/server/mcp/unified_server.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .api_gateway.gateway_router import router as api_router
from .utils.error_handler import handle_error, MCPError
from .utils.performance_metrics import PerformanceMetrics
from .utils.webxos_wallet import WebXOSWallet
from .utils.mcp_server_notes import NotesAPI
from .utils.mcp_server_resources import ResourcesAPI
from .utils.mcp_server_quantum import QuantumAPI
from .api_gateway.service_registry import ServiceRegistry
from .auth.auth_manager import AuthManager
import logging
import asyncio
import psutil
import uvicorn
import json

logger = logging.getLogger("mcp")
app = FastAPI()
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ServiceRegistry()
wallet = WebXOSWallet()
notes_api = NotesAPI()
resources_api = ResourcesAPI()
quantum_api = QuantumAPI()
metrics = PerformanceMetrics()
auth_manager = AuthManager({"address": "0x123", "hash": "abc123", "reputation": 1000})

async def initialize(params):
    logger.info(f"Initializing for user_id: {params.get('user_id')}")
    return {"status": "initialized"}

async def list_tools(params):
    return {"tools": ["wallet", "notes", "resources", "quantum"]}

async def call_tool(params):
    tool_name = params.get("tool_name")
    if tool_name == "wallet":
        return await wallet.connect_wallet(params.get("user_id"))
    elif tool_name == "notes":
        return await notes_api.create_note(params.get("user_id"), "Test", "Test content")
    elif tool_name == "resources":
        return await resources_api.list_resources(params.get("user_id"))
    elif tool_name == "quantum":
        return await quantum_api.simulate_quantum_circuit(params.get("user_id"))
    raise MCPError(code=-32601, message=f"Tool {tool_name} not supported")

async def list_resources(params):
    return await resources_api.list_resources(params.get("user_id"))

async def read_resource(params):
    return await resources_api.read_resource(params.get("user_id"), params.get("uri"))

async def list_prompts(params):
    from .utils.base_prompt import BasePrompt
    prompt = BasePrompt()
    return await prompt.generate_prompt({"method": "test"}, params.get("user_id"))

async def get_prompt(params):
    from .utils.base_prompt import BasePrompt
    prompt = BasePrompt()
    return await prompt.generate_prompt({"method": "test"}, params.get("user_id"), "en")

async def ping(params):
    return {"status": "pong"}

async def create_message(params):
    logger.warning(f"CreateMessageRequest for user_id: {params.get('user_id')}, message: {params.get('message')}")
    return {"status": "message_created"}

async def set_level(params):
    level = params.get("level", "INFO")
    logging.getLogger("mcp").setLevel(level)
    return {"status": f"Logging level set to {level}"}

async def get_system_metrics(params):
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "active_users": 5,
        "balance": 1000.0  # Mock WebXOS balance
    }

async def generate_api_key(params):
    return {"result": await auth_manager.generate_api_key(params.get("user_id"))}

async def import_md(params):
    return await auth_manager.import_md(params.get("user_id"), params.get("md_content"))

async def export_md(params):
    return {"result": await auth_manager.export_md(params.get("user_id"))}

# Register services
registry.register_service("initialize", initialize)
registry.register_service("listTools", list_tools)
registry.register_service("callTool", call_tool)
registry.register_service("listResources", list_resources)
registry.register_service("readResource", read_resource)
registry.register_service("listPrompts", list_prompts)
registry.register_service("getPrompt", get_prompt)
registry.register_service("ping", ping)
registry.register_service("createMessage", create_message)
registry.register_service("setLevel", set_level)
registry.register_service("getSystemMetrics", get_system_metrics)
registry.register_service("generateApiKey", generate_api_key)
registry.register_service("importMd", import_md)
registry.register_service("exportMd", export_md)

@app.post("/mcp/auth")
async def authenticate(request: dict):
    try:
        username = request.get("username")
        password = request.get("password")
        if not username or not password:
            raise MCPError(code=-32602, message="Username and password are required")
        result = await auth_manager.authenticate(username, password)
        return result
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/checklist")
async def get_checklist():
    try:
        checklist = await auth_manager.validate_checklist()
        return {"result": checklist}
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/status")
async def status(request: dict):
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not await auth_manager.verify_token(token):
            raise MCPError(code=-32002, message="Invalid token")
        result = await get_system_metrics({"user_id": "test_user"})
        return {"result": result}
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/api_key")
async def api_key(request: dict):
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not await auth_manager.verify_token(token):
            raise MCPError(code=-32002, message="Invalid token")
        result = await generate_api_key({"user_id": "test_user"})
        return result
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/import")
async def import_endpoint(request: dict):
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not await auth_manager.verify_token(token):
            raise MCPError(code=-32002, message="Invalid token")
        result = await import_md(request.json.get("params", {}))
        return result
    except Exception as e:
        return handle_error(e)

@app.post("/mcp/export")
async def export_endpoint(request: dict):
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not await auth_manager.verify_token(token):
            raise MCPError(code=-32002, message="Invalid token")
        result = await export_md(request.json.get("params", {}))
        return result
    except Exception as e:
        return handle_error(e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
