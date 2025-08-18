import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_model_version_save():
    try:
        response = client.post("/mcp/api/vial/model/version", json={"vial_id": "vial1", "action": "save_version"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_model_version_save", str(e), str(e.__traceback__), sql_statement="INSERT INTO model_versions", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Model version save test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_model_version_invalid_action():
    try:
        response = client.post("/mcp/api/vial/model/version", json={"vial_id": "vial1", "action": "invalid_action"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_model_version_invalid_action", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Model version invalid action test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #model_versioning #sqlite #octokit #neon_mcp
