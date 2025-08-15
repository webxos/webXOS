# main/server/mcp/utils/error_handler.py
import traceback
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger("mcp")

class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)

def handle_error(e: Exception, request_id: Any = None) -> Dict[str, Any]:
    """
    Handles errors and returns a JSON-RPC 2.0 compliant error response with traceback.
    """
    try:
        if isinstance(e, MCPError):
            error_data = {
                "jsonrpc": "2.0",
                "error": {
                    "code": e.code,
                    "message": str(e),
                    "data": {
                        **e.data,
                        "traceback": "".join(traceback.format_tb(e.__traceback__))
                    }
                },
                "id": request_id
            }
            logger.error(f"MCPError: {str(e)}, Traceback: {error_data['error']['data']['traceback']}")
            return error_data
        elif isinstance(e, HTTPException):
            error_data = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": f"HTTP error: {e.detail}",
                    "data": {
                        "status_code": e.status_code,
                        "traceback": "".join(traceback.format_tb(e.__traceback__))
                    }
                },
                "id": request_id
            }
            logger.error(f"HTTPException: {e.detail}, Traceback: {error_data['error']['data']['traceback']}")
            return error_data
        else:
            error_data = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal server error: {str(e)}",
                    "data": {
                        "traceback": "".join(traceback.format_tb(e.__traceback__))
                    }
                },
                "id": request_id
            }
            logger.error(f"Unexpected error: {str(e)}, Traceback: {error_data['error']['data']['traceback']}")
            return error_data
    except Exception as handler_error:
        logger.error(f"Error handler failed: {str(handler_error)}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Error handler failed",
                "data": {"traceback": "".join(traceback.format_tb(handler_error.__traceback__))}
            },
            "id": request_id
        }
