from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tools.auth_tool import AuthTool
from tools.vial_management import VialManagementTool
from tools.health import HealthTool
from tools.blockchain import BlockchainTool
from tools.claude_tool import ClaudeTool
from tools.wallet import WalletTool
from config.config import DatabaseConfig, ServerConfig, limiter, batch_sync_limiter
from lib.notifications import NotificationHandler
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import Dict

app = FastAPI()
app.state.config = ServerConfig()
app.state.db = DatabaseConfig()
app.state.notification_handler = NotificationHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MCPServer:
    def __init__(self):
        self.tools = {
            "authentication": AuthTool(app.state.config),
            "vial_management": VialManagementTool(app.state.db),
            "health": HealthTool(),
            "blockchain": BlockchainTool(app.state.db),
            "claude": ClaudeTool(app.state.db),
            "wallet": WalletTool(app.state.db),
        }

    async def execute(self, request: Dict) -> Dict:
        try:
            if request.get("jsonrpc") != "2.0":
                raise HTTPException(400, "Invalid JSON-RPC request")
            
            method = request.get("method")
            if not method:
                raise HTTPException(400, "Method not specified")
            
            tool_name, method_name = method.split(".", 1) if "." in method else (method, "")
            tool = self.tools.get(tool_name)
            if not tool:
                raise HTTPException(400, "Invalid tool")
            
            params = request.get("params", {})
            params["method"] = method_name
            result = await tool.execute(params)
            
            # Send notification for wallet operations
            if tool_name == "wallet":
                await app.state.notification_handler.send_notification(
                    params.get("user_id"), {"method": method, "params": result.dict()}
                )
            
            return {"jsonrpc": "2.0", "result": result.dict(), "id": request.get("id")}
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"message": str(e)}, "id": request.get("id")}

server = MCPServer()

@app.get("/mcp/health")
async def health_check(request: Request):
    if request.headers.get("X-Forwarded-Proto", "http") != "https":
        raise HTTPException(400, "HTTPS required")
    return {"status": "healthy"}

@app.post("/mcp/execute")
@limiter.limit("10/minute")  # General rate limit
async def execute(request: Request, body: Dict):
    if request.headers.get("X-Forwarded-Proto", "http") != "https":
        raise HTTPException(400, "HTTPS required")
    return await server.execute(body)

@app.post("/mcp/execute/wallet.batchSync")
@batch_sync_limiter
async def batch_sync(request: Request, body: Dict):
    if request.headers.get("X-Forwarded-Proto", "http") != "https":
        raise HTTPException(400, "HTTPS required")
    return await server.execute(body)

@app.websocket("/mcp/notifications")
async def websocket_endpoint(websocket, client_id: str):
    await app.state.notification_handler.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        await app.state.notification_handler.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=app.state.config.host,
        port=app.state.config.port,
        ssl_keyfile=app.state.config.ssl_key_path,
        ssl_certfile=app.state.config.ssl_cert_path
    )
