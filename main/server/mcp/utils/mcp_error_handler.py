# main/server/mcp/utils/mcp_error_handler.py
from pydantic import BaseModel

class MCPError(Exception):
    def __init__(self, code: int, message: str, data: dict = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(self.message)

class MCPErrorResponse(BaseModel):
    code: int
    message: str
    data: dict

def handle_mcp_error(error: MCPError) -> dict:
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": error.code,
            "message": error.message,
            "data": error.data
        }
    }

# Standard MCP error codes
MCP_ERROR_CODES = {
    "INVALID_REQUEST": -32600,
    "METHOD_NOT_FOUND": -32601,
    "INVALID_PARAMS": -32602,
    "INTERNAL_ERROR": -32603,
    "UNAUTHORIZED": -32001,
    "FORBIDDEN": -32003,
    "RATE_LIMIT_EXCEEDED": -32029
}
