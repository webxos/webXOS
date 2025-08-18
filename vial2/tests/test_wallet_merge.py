import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..wallet.wallet_merge import merge_wallets
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_wallet_merge():
    try:
        wallet_addresses = ["0x1234567890abcdef1234567890abcdef12345678", "0xabcdef1234567890abcdef1234567890abcdef12"]
        result = await merge_wallets(1, wallet_addresses)
        assert result["status"] == "success"
        assert "merged_hash" in result
    except Exception as e:
        error_logger.log_error("test_wallet_merge", str(e), str(e.__traceback__))
        logger.error(f"Wallet merge test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #wallet_merge #neon_mcp
