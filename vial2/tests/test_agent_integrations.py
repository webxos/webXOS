import pytest
from fastapi.testclient import TestClient
from ..main import app
import logging
import os

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_agent_integration():
    try:
        # Mock environment variable for testing
        os.environ["GROK_API_KEY"] = "test_key"
        response = client.post("/mcp/api/vial/agent", json={"agent_type": "grok", "message": {"content": "Test message"}})
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["status"] == "success"
        assert "Response from grok" in data["result"]["data"]
        logger.info("Agent integration test passed")
    except Exception as e:
        logger.error(f"Agent integration test failed: {str(e)}")
        raise
    finally:
        if "GROK_API_KEY" in os.environ:
            del os.environ["GROK_API_KEY"]

# xAI Artifact Tags: #vial2 #tests #mcp #agent #integration #neon_mcp
