from fastapi import Request
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class HTTPTransport:
    async def handle_request(self, request: Request):
        try:
            body = await request.json()
            logger.info(f"Received HTTP request: {body}")
            return {"jsonrpc": "2.0", "result": {"status": "success", "data": body}}
        except Exception as e:
            error_logger.log_error("http_transport", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"HTTP transport failed: {str(e)}")
            return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

http_transport = HTTPTransport()

# xAI Artifact Tags: #vial2 #mcp #transport #http #neon_mcp
