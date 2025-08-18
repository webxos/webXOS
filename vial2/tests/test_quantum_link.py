import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_quantum_link_train():
    try:
        response = client.post("/mcp/api/quantum_link", json={"type": "train", "vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "loss" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_quantum_link_train", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET quantum_state", sql_error_code=None, params={"type": "train"})
        logger.error(f"Quantum link train test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_quantum_link_invalid_op():
    try:
        response = client.post("/mcp/api/quantum_link", json={"type": "invalid_op", "vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_quantum_link_invalid_op", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"type": "invalid_op"})
        logger.error(f"Quantum link invalid operation test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #quantum_link #pytorch #octokit #sqlite #neon_mcp
