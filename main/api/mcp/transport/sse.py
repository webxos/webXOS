from fastapi import APIRouter, Response
from fastapi_socketio import SocketManager
from ...utils.logging import log_error, log_info
from ...mcp.server import MCPServer

router = APIRouter()
sio = SocketManager()
mcp_server = MCPServer()

@sio.on('connect')
async def handle_connect(sid, environ):
    log_info(f"SSE connected: {sid}")

@sio.on('disconnect')
async def handle_disconnect(sid):
    log_info(f"SSE disconnected: {sid}")

@sio.on('mcp_request')
async def handle_sse_request(sid, data):
    try:
        method = data.get('method')
        params = data.get('params', {})
        request_id = data.get('id')
        if method == "initialize":
            result = await mcp_server.initialize(params)
        elif method == "initialized":
            await mcp_server.initialized(params)
            await sio.emit('mcp_response', {"jsonrpc": "2.0", "id": request_id}, to=sid)
            return
        elif method == "tools/list":
            result = await mcp_server.list_tools()
        elif method == "tools/call":
            result = await mcp_server.call_tool(params.get("name"), params.get("arguments", {}))
        elif method == "resources/list":
            result = await mcp_server.list_resources()
        elif method == "resources/read":
            result = await mcp_server.read_resource(params.get("uri"))
        elif method == "prompts/list":
            result = await mcp_server.list_prompts()
        elif method == "prompts/get":
            result = await mcp_server.get_prompt(params.get("name"), params.get("arguments", []))
        else:
            log_error(f"Unknown SSE method: {method}")
            await sio.emit('mcp_response', {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown method: {method}"}, "id": request_id}, to=sid)
            return
        log_info(f"SSE {method} executed for {sid}")
        await sio.emit('mcp_response', {"jsonrpc": "2.0", "result": result, "id": request_id}, to=sid)
    except Exception as e:
        log_error(f"SSE request failed: {str(e)}")
        await sio.emit('mcp_response', {"jsonrpc": "2.0", "error": {"code": -32601, "message": str(e)}, "id": request_id}, to=sid)
