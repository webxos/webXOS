# main/server/mcp/utils/error_handler.py
import traceback
from typing import Any

class MCPError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

def handle_error(error: Exception) -> dict:
    error_dict = {
        "jsonrpc": "2.0",
        "error": {
            "code": -32603 if not isinstance(error, MCPError) else error.code,
            "message": str(error),
            "traceback": traceback.format_exc()
        },
        "id": None
    }
    return error_dict
