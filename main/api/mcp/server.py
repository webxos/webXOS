from fastapi import APIRouter, Depends
from pydantic import BaseModel
from utils.auth import verify_token

class MCPRequest(BaseModel):
    jsonrpc: str
    method: str
    params: dict
    id: int

class MCPServer:
    def __init__(self):
        self.router = APIRouter()

        @self.router.post("/mcp")
        async def mcp_endpoint(request: MCPRequest, token: str = Depends(verify_token)):
            if request.jsonrpc != "2.0":
                return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request.id}
            method = request.method
            params = request.params
            if method == "tools/list":
                return {"jsonrpc": "2.0", "result": ["get_wallet_info", "generate_credentials"], "id": request.id}
            elif method == "prompts/get":
                return {"jsonrpc": "2.0", "result": f"Response for {params.get('name', 'unknown')}", "id": request.id}
            elif method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "serverName": "webxos-mcp-gateway",
                        "serverVersion": "2.7.8",
                        "protocolVersion": "2024-11-05",
                        "capabilities": ["tools", "resources", "prompts", "tasks", "notifications"]
                    },
                    "id": request.id
                }
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": request.id}
