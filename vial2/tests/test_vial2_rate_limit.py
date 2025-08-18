import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_rate_limit_set():
    try:
        response = client.post("/mcp/api/vial/rate_limit", json={"vial_id": "vial1", "limit": 50, "window": 1800}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"]["limit"] == 50
        assert response.json()["result"]["data"]["window"] == 1800
    except Exception as e:
        error_logger.log_error("test_rate_limit_set", str(e), str(e.__traceback__), sql_statement="INSERT OR REPLACE INTO rate_limits", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Rate limit set test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_rate_limit_missing_vial():
    try:
        response = client.post("/mcp/api/vial/rate_limit", json={"limit": 50, "window": 1800}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_rate_limit_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Rate limit missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #rate_limit #sqlite #octokit #neon_mcp
