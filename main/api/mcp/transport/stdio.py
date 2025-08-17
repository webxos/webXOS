import json
import sys
from typing import Dict, Any
from ...utils.logging import log_error, log_info
from ..server import MCPServer

class StdioTransport:
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server

    async def handle_message(self, message: str):
        try:
            request = json.loads(message)
            if request.get("jsonrpc") != "2.0":
                raise ValueError("Invalid JSON-RPC version")
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            response = await self.dispatch(method, params, request_id)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            log_info(f"Stdio request processed: {method}")
        except Exception as e:
            log_error(f"Stdio error: {str(e)}")
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": str(e)},
                "id": request_id
            }) + "\n")
            sys.stdout.flush()

    async def dispatch(self, method: str, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        try:
            if method == "initialize":
                result = await self.mcp_server.initialize(params)
            elif method == "initialized":
                await self.mcp_server.initialized(params)
                return {"jsonrpc": "2.0", "id": request_id}
            elif method == "tools/list":
                result = await self.mcp_server.list_tools()
            elif method == "tools/call":
                result = await self.mcp_server.call_tool(params.get("name"), params.get("arguments", {}))
            elif method == "resources/list":
                result = await self.mcp_server.list_resources()
            elif method == "resources/read":
                result = await self.mcp_server.read_resource(params.get("uri"))
            elif method == "prompts/list":
                result = await self.mcp_server.list_prompts()
            elif method == "prompts/get":
                result = await self.mcp_server.get_prompt(params.get("name"), params.get("arguments", []))
            else:
                raise ValueError(f"Unknown method: {method}")
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except Exception as e:
            log_error(f"Dispatch error for {method}: {str(e)}")
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": str(e)}, "id": request_id}

    def run(self):
        log_info("Starting Stdio transport")
        while True:
            message = sys.stdin.readline().strip()
            if not message:
                break
            asyncio.run(self.handle_message(message))
