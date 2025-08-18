import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging
import json

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_wallet_import():
    try:
        import_data = json.dumps({"wallet_address": "0x1234567890abcdef1234567890abcdef12345678", "balance": 100.0})
        response = client.post("/mcp/api/vial/import", json={"vial_id": "vial1", "import_data": import_data}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["address"] == "0x1234567890abcdef1234567890abcdef12345678"
    except Exception as e:
        error_logger.log_error("test_wallet_import", str(e), str(e.__traceback__), sql_statement="INSERT INTO wallets", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Wallet import test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_wallet_import_invalid():
    try:
        response = client.post("/mcp/api/vial/import", json={"vial_id": "vial1", "import_data": "invalid_json"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_wallet_import_invalid", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Wallet import invalid test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #import #sqlite #octokit #neon_mcp
