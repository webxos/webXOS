import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_git_status():
    try:
        response = client.post("/mcp/api/vial/git", json={"vial_id": "vial1", "command": "git status"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_git_status", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET git_state", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Git status test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_git_invalid_command():
    try:
        response = client.post("/mcp/api/vial/git", json={"vial_id": "vial1", "command": "git invalid"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_git_invalid_command", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Git invalid command test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #git #sqlite #octokit #neon_mcp
