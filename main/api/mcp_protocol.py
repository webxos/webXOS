import json
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
from ..utils.logging import log_error

router = APIRouter(prefix="/v1/mcp", tags=["MCP Protocol"])

# MCP JSON-RPC 2.0 Request Schema
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: int | str | None = None

# MCP JSON-RPC 2.0 Response Schema
class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any | None = None
    error: Dict[str, Any] | None = None
    id: int | str | None = None

# MCP Error Codes
MCP_ERROR_CODES = {
    -32700: {"code": -32700, "message": "Parse error"},
    -32600: {"code": -32600, "message": "Invalid Request"},
    -32601: {"code": -32601, "message": "Method not found"},
    -32602: {"code": -32602, "message": "Invalid params"},
    -32603: {"code": -32603, "message": "Internal error"}
}

@router.post("/rpc")
async def mcp_endpoint(request: MCPRequest):
    """Handle MCP JSON-RPC 2.0 requests."""
    try:
        # Validate MCP protocol version
        if request.jsonrpc != "2.0":
            return MCPResponse(
                jsonrpc="2.0",
                error=MCP_ERROR_CODES[-32600],
                id=request.id
            )

        # Supported MCP methods
        methods = {
            "initialize": lambda params: {
                "capabilities": {
                    "tools": ["completion", "wallet", "quantum_link"],
                    "resources": ["mongodb", "sqlite"],
                    "prompts": ["claude", "grok", "gemini"],
                    "logging": True
                }
            },
            "initialized": lambda params: {"status": "ok"},
            "ping": lambda params: {"status": "pong"},
            "progress": lambda params: {"progress": params.get("progress", 0)},
            "cancelled": lambda params: {"status": "cancelled"}
        }

        if request.method not in methods:
            return MCPResponse(
                jsonrpc="2.0",
                error=MCP_ERROR_CODES[-32601],
                id=request.id
            )

        try:
            result = methods[request.method](request.params)
            return MCPResponse(jsonrpc="2.0", result=result, id=request.id)
        except Exception as e:
            log_error(f"MCP method {request.method} failed: {str(e)}")
            return MCPResponse(
                jsonrpc="2.0",
                error={**MCP_ERROR_CODES[-32603], "data": str(e)},
                id=request.id
            )

    except ValidationError as e:
        log_error(f"MCP request validation failed: {str(e)}")
        return MCPResponse(
            jsonrpc="2.0",
            error={**MCP_ERROR_CODES[-32602], "data": str(e)},
            id=request.id
        )
    except Exception as e:
        log_error(f"MCP internal error: {str(e)}")
        return MCPResponse(
            jsonrpc="2.0",
            error={**MCP_ERROR_CODES[-32603], "data": str(e)},
            id=request.id
        )
