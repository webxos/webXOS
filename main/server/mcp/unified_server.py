# main/server/mcp/unified_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api_gateway.gateway_router import router as api_router
from .utils.error_handler import handle_error, MCPError
from .utils.performance_metrics import PerformanceMetrics
from .utils.webxos_wallet import WebXOSWallet
from .utils.mcp_server_notes import NotesAPI
from .utils.mcp_server_resources import ResourcesAPI
from .utils.mcp_server_quantum import QuantumAPI
from .api_gateway.service_registry import ServiceRegistry
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
        "active_users": 5,  # Placeholder; replace with actual count
    }

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

@app.post("/mcp")
async def mcp_endpoint(request: dict):
    try:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not method:
            raise MCPError(code=-32600, message="Invalid Request")
        result = await registry.dispatch(method, params, request_id)
        return result
    except Exception as e:
        return handle_error(e, request.get("id"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
