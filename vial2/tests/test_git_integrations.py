import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_git_model_commit():
    try:
        response = client.post("/mcp/api/vial/git/model", json={"vial_id": "vial1", "action": "commit_model"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_git_model_commit", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET git_model_state", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Git model commit test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_git_model_invalid_action():
    try:
        response = client.post("/mcp/api/vial/git/model", json={"vial_id": "vial1", "action": "invalid_action"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_git_model_invalid_action", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Git model invalid action test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #git #integration #sqlite #octokit #neon_mcp
