from fastapi import Request, WebSocket
from ..security.audit import security_audit
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class SecureTransport:
    async def secure_http(self, request: Request):
        try:
            if not await security_audit.check_input_sanitization(await request.json()):
                raise ValueError("Insecure HTTP request")
            logger.info("Secured HTTP request processed")
            return {"jsonrpc": "2.0", "result": {"status": "success", "data": "Secured"}}
        except Exception as e:
            error_logger.log_error("secure_http", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Secure HTTP failed: {str(e)}")
            raise

    async def secure_websocket(self, websocket: WebSocket):
        try:
            await websocket.accept()
            data = await websocket.receive_text()
            if not await security_audit.check_input_sanitization(data):
                raise ValueError("Insecure WebSocket message")
            await websocket.send_text(json.dumps({"jsonrpc": "2.0", "result": {"status": "success", "data": data}}))
            logger.info("Secured WebSocket message processed")
        except Exception as e:
            error_logger.log_error("secure_websocket", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Secure WebSocket failed: {str(e)}")
            await websocket.close()

secure_transport = SecureTransport()

# xAI Artifact Tags: #vial2 #mcp #transport #secure #neon_mcp
