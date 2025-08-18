import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_endpoint():
    try:
        response = client.get("/mcp/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    except Exception as e:
        error_logger.log_error("test_health_endpoint", str(e), str(e.__traceback__))
        logger.error(f"Health endpoint test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_command_endpoint():
    try:
        response = client.post("/mcp/api/endpoints", json={"command": "test"})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_command_endpoint", str(e), str(e.__traceback__))
        logger.error(f"Command endpoint test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #endpoints #neon_mcp
