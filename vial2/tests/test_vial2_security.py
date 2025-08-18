import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_security_policy_set():
    try:
        response = client.post("/mcp/api/vial/security", json={"vial_id": "vial1", "policy": "restrict_access"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["security_policy"] == "restrict_access"
    except Exception as e:
        error_logger.log_error("test_security_policy_set", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET security_policy", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Security policy set test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_security_policy_invalid():
    try:
        response = client.post("/mcp/api/vial/security", json={"vial_id": "vial1", "policy": "invalid_policy"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_security_policy_invalid", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Security policy invalid test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #security #sqlite #octokit #neon_mcp
