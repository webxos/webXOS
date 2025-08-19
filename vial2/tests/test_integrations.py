import pytest
from mcp.client import mcp_client
from mcp.server import mcp_server
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_end_to_end():
    try:
        await mcp_server.start()
        await mcp_client.connect()
        response = await mcp_client.send_request("initialize", {"id": "1"})
        assert response["jsonrpc"] == "2.0"
        logger.info("End-to-end integration test passed")
    except Exception as e:
        logger.error(f"End-to-end integration test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #integration #neon_mcp
