import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_error_metrics():
    try:
        response = client.get("/mcp/api/vial/error/metrics", headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_error_metrics", str(e), str(e.__traceback__), sql_statement="SELECT event_type FROM vial_logs", sql_error_code=None, params={})
        logger.error(f"Error metrics test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #error_metrics #sqlite #octokit #neon_mcp
