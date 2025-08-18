import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_git_command_status():
    try:
        response = client.post("/mcp/api/git", json={"type": "status"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_git_command_status", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"type": "status"})
        logger.error(f"Git command status test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_git_command_invalid():
    try:
        response = client.post("/mcp/api/git", json={"type": "invalid"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32603
        assert response.json()["error"]["message"] == "Invalid Git command"
    except Exception as e:
        error_logger.log_error("test_git_command_invalid", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"type": "invalid"})
        logger.error(f"Git command invalid test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #git #octokit #sqlite #neon_mcp
