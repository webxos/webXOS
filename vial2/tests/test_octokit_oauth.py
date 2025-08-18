import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_web_flow_url
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_octokit_oauth_web_flow():
    try:
        result = await get_web_flow_url()
        assert "url" in result
        assert "state" in result
        assert "github.com/login/oauth/authorize" in result["url"]
    except Exception as e:
        error_logger.log_error("test_octokit_web_flow", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Octokit OAuth web flow test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_octokit_oauth_endpoint():
    try:
        response = client.post("/mcp/api/auth", json={"code": "test_code", "redirect_uri": "http://localhost:8000/callback"})
        assert response.status_code == 400  # Expect failure due to invalid code
        assert "error" in response.json()
    except Exception as e:
        error_logger.log_error("test_octokit_oauth_endpoint", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"code": "test_code"})
        logger.error(f"Octokit OAuth endpoint test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #octokit #oauth #sqlite #neon_mcp
