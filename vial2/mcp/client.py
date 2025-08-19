from mcp.server.stdio import stdio_client
from mcp.error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.client = stdio_client()

    async def connect(self):
        try:
            await self.client.connect()
            logger.info("MCP client connected")
        except Exception as e:
            error_logger.log_error("mcp_client_connect", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"MCP client connection failed: {str(e)}")
            raise

    async def send_request(self, method, params):
        try:
            response = await self.client.send({"jsonrpc": "2.0", "method": method, "params": params, "id": "1"})
            logger.info(f"Client sent request {method}")
            return response
        except Exception as e:
            error_logger.log_error("mcp_client_request", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Client request failed: {str(e)}")
            raise

mcp_client = MCPClient()

# xAI Artifact Tags: #vial2 #mcp #client #mcp_compliance #neon_mcp
