import pytest
from mcp.server import MCPServer
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_mcp_server():
    try:
        server = MCPServer()
        await server.start()
        result = await server.handle_request("initialize", {"id": "1"})
        assert result["jsonrpc"] == "2.0"
        logger.info("MCP server compliance test passed")
    except Exception as e:
        logger.error(f"MCP server compliance test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_jsonrpc_methods():
    try:
        server = MCPServer()
        result = await server.handle_request("tools/list", {"id": "2"})
        assert "tools" in result["result"]
        logger.info("JSON-RPC methods test passed")
    except Exception as e:
        logger.error(f"JSON-RPC methods test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #compliance #neon_mcp
