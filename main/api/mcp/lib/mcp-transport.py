from pydantic import BaseModel
from typing import Dict, Any, Callable
import logging

logger = logging.getLogger("mcp.transport")
logger.setLevel(logging.INFO)

class MCPTransport:
    def __init__(self, handler: Callable):
        self.handler = handler

    async def handle(self, request: Any) -> Any:
        if request.jsonrpc != "2.0":
            logger.error("Invalid JSON-RPC version")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid JSON-RPC version"},
                "id": request.id
            }
        return await self.handler(request)
