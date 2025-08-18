import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_vial_wallet_export():
    try:
        response = client.post("/mcp/api/vial/export", json={"vial_id": "vial1", "wallet_address": "0x1234567890abcdef1234567890abcdef12345678"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "export_data" in response.json()["result"]
        assert response.json()["result"]["export_data"]["vial_id"] == "vial1"
    except Exception as e:
        error_logger.log_error("test_vial_wallet_export", str(e), str(e.__traceback__), sql_statement="SELECT wallet_address, balance FROM wallets", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial wallet export test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_vial_wallet_export_invalid():
    try:
        response = client.post("/mcp/api/vial/export", json={"vial_id": "vial1", "wallet_address": "invalid_address"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_vial_wallet_export_invalid", str(e), str(e.__traceback__), sql_statement="SELECT wallet_address, balance FROM wallets", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Vial wallet export invalid test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #vial #export #sqlite #octokit #neon_mcp
