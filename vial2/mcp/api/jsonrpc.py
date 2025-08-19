from fastapi import HTTPException
import json
from mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class JSONRPCHandler:
    def __init__(self):
        self.methods = {
            "initialize": self.initialize,
            "tools/list": self.list_tools,
            "resources/list": self.list_resources
        }

    async def handle(self, method, params):
        try:
            if method not in self.methods:
                raise ValueError(f"Method {method} not found")
            result = await self.methods[method](params)
            return {"jsonrpc": "2.0", "result": result, "id": params.get("id")}
        except Exception as e:
            error_logger.log_error("jsonrpc_handle", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"JSON-RPC handling failed: {str(e)}")
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": str(e)}, "id": params.get("id")}

    async def initialize(self, params):
        return {"capabilities": {"tools": ["/vial/train", "/vial/sync"], "resources": ["vial://config"]}}

    async def list_tools(self, params):
        return ["/vial/train", "/vial/sync", "/vial/quantum"]

    async def list_resources(self, params):
        return ["vial://config", "vial://wallet", "vial://status"]

jsonrpc_handler = JSONRPCHandler()

# xAI Artifact Tags: #vial2 #mcp #api #jsonrpc #mcp_compliance #neon_mcp
