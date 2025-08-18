import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_quantum_link():
    try:
        response = client.post("/mcp/api/vial/quantum/link", json={"vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["quantum_state"]["linked"] is True
    except Exception as e:
        error_logger.log_error("test_quantum_link", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET quantum_linked", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Quantum link test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_quantum_link_missing_vial():
    try:
        response = client.post("/mcp/api/vial/quantum/link", json={}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_quantum_link_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Quantum link missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #quantum #sqlite #octokit #neon_mcp
