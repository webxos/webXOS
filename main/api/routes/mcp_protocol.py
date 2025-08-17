from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ..mcp.server import MCPServer
from ..mcp_schemas import MCPRequest, MCPResponse, MCPInitializeRequest, MCPInitializedParams

router = APIRouter()
mcp_server = MCPServer()

@router.post("/mcp")
async def handle_mcp_request(request: MCPRequest, user_id: str = Depends(verify_token)):
    try:
        method = request.method
        params = request.params or {}
        request_id = request.id
        if method == "initialize":
            result = await mcp_server.initialize(params)
        elif method == "initialized":
            await mcp_server.initialized(params)
            return MCPResponse(jsonrpc="2.0", id=request_id)
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
            log_error(f"Unknown MCP method: {method}")
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
        log_info(f"MCP {method} executed for user {user_id}")
        return MCPResponse(jsonrpc="2.0", result=result, id=request_id)
    except Exception as e:
        log_error(f"MCP request failed: {str(e)}")
        return MCPResponse(jsonrpc="2.0", error={"code": -32601, "message": str(e)}, id=request_id)
