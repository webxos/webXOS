from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from tools.auth_tool import AuthenticationTool
from tools.vial_management import VialManagementTool
from tools.health import HealthTool
from tools.blockchain import BlockchainTool
from tools.claude_tool import ClaudeTool
from lib.mcp_transport import MCPTransport
from lib.notifications import NotificationHandler
from config.config import DatabaseConfig
import logging

app = FastAPI()
logger = logging.getLogger("mcp")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: int

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Any = None
    id: int

class MCPServer:
    def __init__(self):
        self.db = DatabaseConfig()
        self.notification_handler = NotificationHandler()
        self.tools = {
            "authentication": AuthenticationTool(self.db),
            "vial-management": VialManagementTool(self.db),
            "health": HealthTool(self.db, self.tools),
            "blockchain": BlockchainTool(self.db),
            "claude": ClaudeTool(self.db)
        }
        self.transport = MCPTransport(self.handle_request)

    async def start(self):
        await self.db.connect()
        logger.info("MCP Server started on port 8000")

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        try:
            tool_name, *method_parts = request.method.split(".")
            if tool_name not in self.tools:
                raise HTTPException(404, f"Unknown tool: {tool_name}")
            tool = self.tools[tool_name]
            result = await tool.execute(request.params)
            # Send notification for Claude code execution
            if tool_name == "claude":
                await self.notification_handler.send_notification(
                    request.params.get("user_id", "default"),
                    {"jsonrpc": "2.0", "method": "claude.executionComplete", "params": result}
                )
            return MCPResponse(id=request.id, result=result, error=None)
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return MCPResponse(
                id=request.id,
                error={"code": -32000, "message": str(e)},
                result=None
            )

server = MCPServer()

@app.on_event("startup")
async def startup_event():
    await server.start()

@app.post("/mcp/execute")
async def execute(request: MCPRequest):
    return await server.transport.handle(request)

@app.get("/mcp/health")
async def health_check():
    return await server.tools["health"].execute({})

@app.websocket("/mcp/notifications")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await server.notification_handler.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await server.notification_handler.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
