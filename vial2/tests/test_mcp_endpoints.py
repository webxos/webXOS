import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_mcp_status():
    try:
        response = client.post("/mcp/api/vial/mcp/status", json={}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_mcp_status", str(e), str(e.__traceback__), sql_statement="SELECT vial_id, status FROM vials", sql_error_code=None, params={})
        logger.error(f"MCP status test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_mcp_connect():
    try:
        response = client.post("/mcp/api/vial/mcp/connect", json={"vial_id": "vial1", "server": "default"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["mcp_server"] == "default"
    except Exception as e:
        error_logger.log_error("test_mcp_connect", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET mcp_server", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"MCP connect test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #endpoints #sqlite #octokit #neon_mcp
