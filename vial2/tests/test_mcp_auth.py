import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_mcp_auth_token():
    try:
        response = client.post("/mcp/api/vial/mcp/auth/token", json={"vial_id": "vial1", "mcp_key": "test_key"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["mcp_key"] == "test_key"
    except Exception as e:
        error_logger.log_error("test_mcp_auth_token", str(e), str(e.__traceback__), sql_statement="INSERT INTO mcp_tokens", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"MCP auth token test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_mcp_auth_token_missing_vial():
    try:
        response = client.post("/mcp/api/vial/mcp/auth/token", json={"mcp_key": "test_key"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_mcp_auth_token_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"MCP auth token missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #auth #sqlite #octokit #neon_mcp
