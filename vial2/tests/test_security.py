import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_oauth_authentication():
    try:
        response = client.get("/mcp/auth/relay_check", headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_oauth_auth", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"OAuth test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_oauth_invalid_token():
    try:
        response = client.get("/mcp/auth/relay_check")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == -32000
    except Exception as e:
        error_logger.log_error("test_oauth_invalid", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Invalid OAuth test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #security #oauth #neon_mcp
