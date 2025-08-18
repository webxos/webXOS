import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_wallet_export():
    try:
        response = client.post("/mcp/api/vial/wallet/export", json={"vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "data" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_wallet_export", str(e), str(e.__traceback__), sql_statement="INSERT INTO wallet_exports", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Wallet export test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_wallet_export_missing_vial():
    try:
        response = client.post("/mcp/api/vial/wallet/export", json={}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_wallet_export_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Wallet export missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #wallet #export #sqlite #octokit #neon_mcp
