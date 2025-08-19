from .http_transport import http_transport
from .websocket_transport import websocket_transport
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class TransportManager:
    def __init__(self):
        self.transports = {"http": http_transport, "websocket": websocket_transport}

    async def route_request(self, request_type: str, *args, **kwargs):
        try:
            transport = self.transports.get(request_type.lower())
            if not transport:
                raise ValueError("Unsupported transport type")
            return await transport.handle_request(*args) if request_type == "http" else await transport.handle_connection(*args)
        except Exception as e:
            error_logger.log_error("transport_manager_route", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Transport routing failed: {str(e)}")
            raise

transport_manager = TransportManager()

# xAI Artifact Tags: #vial2 #mcp #transport #manager #neon_mcp
