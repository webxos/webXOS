import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_sync_state():
    try:
        response = client.post("/mcp/api/vial/sync/state", json={"vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_sync_state", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET last_sync", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Sync state test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_sync_state_missing_vial():
    try:
        response = client.post("/mcp/api/vial/sync/state", json={}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_sync_state_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Sync state missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #sync #sqlite #octokit #neon_mcp
