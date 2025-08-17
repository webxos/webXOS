from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from tools.auth_tool import AuthenticationTool
from lib.mcp_transport import MCPTransport
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
        self.tools = {
            "authentication": AuthenticationTool(self.db),
            # Add vial-management tool later
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
