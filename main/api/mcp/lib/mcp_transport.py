from pydantic import BaseModel
from typing import Dict, Any, Callable
from lib.errors import ValidationError
from lib.mcp_protocol import MCPRequest, MCPResponse
import logging
import json

logger = logging.getLogger("mcp.transport")
logger.setLevel(logging.INFO)

class MCPTransport:
    def __init__(self, handler: Callable):
        self.handler = handler

    async def handle(self, request: Any) -> Any:
        try:
            # Validate JSON-RPC request
            mcp_request = MCPRequest(**request.dict())
            logger.info(f"Received MCP request: {mcp_request.method}")

            # Handle Claude code execution requests
            if mcp_request.method.startswith("claude."):
                # Ensure code parameter is present for Claude methods
                if "code" not in mcp_request.params:
                    raise ValidationError("Claude methods require a 'code' parameter")

            # Process request through handler
            result = await self.handler(mcp_request)
            response = MCPResponse(id=mcp_request.id, result=result, error=None)
            logger.info(f"Request processed successfully: {mcp_request.method}")
            return response.dict(exclude_none=True)

        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return MCPResponse(
                id=request.id if hasattr(request, 'id') else None,
                error={"code": -32602, "message": str(e)},
                result=None
            ).dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Transport error: {str(e)}")
            return MCPResponse(
                id=request.id if hasattr(request, 'id') else None,
                error={"code": -32000, "message": str(e)},
                result=None
            ).dict(exclude_none=True)
