from fastapi import WebSocket
from ..error_logging.error_log import error_logger
import logging
import json

logger = logging.getLogger(__name__)

class WebSocketTransport:
    async def handle_connection(self, websocket: WebSocket):
        try:
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data}")
                await websocket.send_text(json.dumps({"jsonrpc": "2.0", "result": {"status": "success", "data": data}}))
        except Exception as e:
            error_logger.log_error("websocket_transport", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"WebSocket transport failed: {str(e)}")
            await websocket.close()

websocket_transport = WebSocketTransport()

# xAI Artifact Tags: #vial2 #mcp #transport #websocket #neon_mcp
