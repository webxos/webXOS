import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_vial_manage_start():
    try:
        response = client.post("/mcp/api/vial/manage", json={"type": "start", "vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["status"] == "running"
    except Exception as e:
        error_logger.log_error("test_vial_manage_start", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET status", sql_error_code=None, params={"type": "start"})
        logger.error(f"Vial manage start test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_vial_proof_of_work():
    try:
        response = client.post("/mcp/api/vial/pow", json={"vial_id": "vial1", "difficulty": 2, "wallet_address": "0x1234567890abcdef1234567890abcdef12345678"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "pow" in response.json()["result"]
        assert response.json()["result"]["wallet_updated"]
    except Exception as e:
        error_logger.log_error("test_vial_proof_of_work", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET pow_result", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial proof of work test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_vial_sync():
    try:
        response = client.post("/mcp/api/vial/sync", json={"vial_id": "vial1", "node_id": "node1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_vial_sync", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET status", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial sync test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_vial_wallet_bridge():
    try:
        response = client.post("/mcp/api/vial/wallet_bridge", json={"vial_id": "vial1", "wallet_address": "0x1234567890abcdef1234567890abcdef12345678", "action": "link"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_vial_wallet_bridge", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET wallet_address", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial wallet bridge test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_vial_auth_verify():
    try:
        response = client.post("/mcp/api/vial/auth/verify", json={"vial_id": "vial1", "wallet_address": "0x1234567890abcdef1234567890abcdef12345678"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["verified"]
    except Exception as e:
        error_logger.log_error("test_vial_auth_verify", str(e), str(e.__traceback__), sql_statement="SELECT wallet_address FROM vials", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial auth verify test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #vial #sqlite #octokit #neon_mcp
