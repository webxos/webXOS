# main/server/mcp/utils/error_handler.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from ..utils.mcp_error_handler import MCPError
import logging
import re
import json

logger = logging.getLogger("mcp")

class ErrorHandler:
    def __init__(self):
        self.suspicious_patterns = [
            r"secret_key|api_key|token|password|credential",  # Secret keywords
            r"<!--\s*hidden:\s*\".*?\"\s*-->",  # Hidden HTML comments (prompt injection)
            r"exec\(|eval\(|system\(",  # Code execution attempts
            r"private\s*repo|all\s*repos"  # Attempts to access private repos
        ]
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.suspicious_patterns]

    async def detect_prompt_injection(self, content: str) -> bool:
        try:
            for pattern in self.patterns:
                if pattern.search(content):
                    logger.warning(f"Potential prompt injection detected: {content[:100]}...")
                    return True
            return False
        except Exception as e:
            logger.error(f"Prompt injection detection failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to process content: {str(e)}")

    async def handle_request(self, request: Request, call_next):
        try:
            # Inspect request body for prompt injection
            if request.method in ["POST", "PUT"]:
                body = await request.body()
                if body:
                    content = body.decode("utf-8")
                    if await self.detect_prompt_injection(content):
                        raise MCPError(
                            code=-32004,
                            message="Potential prompt injection detected in request"
                        )
            
            response = await call_next(request)
            return response
        except MCPError as e:
            logger.error(f"MCP Error: {e.message} (code: {e.code})")
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "error": {"code": e.code, "message": e.message}, "id": None}
            )
        except HTTPException as e:
            logger.error(f"HTTP Error: {str(e)}")
            return JSONResponse(
                status_code=e.status_code,
                content={"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e.detail)}, "id": None}
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {str(e)}"}, "id": None}
            )
