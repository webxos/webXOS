# main/server/mcp/api_gateway/gateway_router.py
from fastapi import FastAPI
from .service_registry import ServiceRegistry
from ..utils.error_handler import handle_error

app = FastAPI()
registry = ServiceRegistry()

@app.post("/mcp")
async def route_request(data: dict):
    try:
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")
        return await registry.dispatch(method, params, request_id)
    except Exception as e:
        return handle_error(e)
