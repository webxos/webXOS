import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_wallet_validate():
    try:
        response = client.post("/mcp/api/wallet_ops", json={"type": "validate", "address": "0x1234567890abcdef1234567890abcdef12345678"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_wallet_validate", str(e), str(e.__traceback__), sql_statement="SELECT * FROM wallets", sql_error_code=None, params={"type": "validate"})
        logger.error(f"Wallet validate test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_wallet_update_balance():
    try:
        response = client.post("/mcp/api/wallet_ops", json={"type": "update_balance", "address": "0x1234567890abcdef1234567890abcdef12345678", "amount": 10.0}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "new_balance" in response.json()["result"]
    except Exception as e:
        error_logger.log_error("test_wallet_update_balance", str(e), str(e.__traceback__), sql_statement="UPDATE wallets SET balance", sql_error_code=None, params={"type": "update_balance"})
        logger.error(f"Wallet update balance test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_wallet_invalid_op():
    try:
        response = client.post("/mcp/api/wallet_ops", json={"type": "invalid_op", "address": "0x1234567890abcdef1234567890abcdef12345678"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_wallet_invalid_op", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"type": "invalid_op"})
        logger.error(f"Wallet invalid operation test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #wallet #octokit #sqlite #neon_mcp
