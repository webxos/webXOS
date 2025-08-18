import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_auth_endpoint():
    try:
        response = client.post("/mcp/api/endpoints", json={"command": "test"}, headers={"Authorization": "Bearer invalid_token"})
        assert response.status_code == 401
    except Exception as e:
        error_logger.log_error("test_auth", str(e), str(e.__traceback__))
        logger.error(f"Auth test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #auth #neon_mcp
