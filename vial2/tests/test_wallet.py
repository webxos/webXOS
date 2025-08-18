import pytest
from ..wallet.wallet_manager import import_wallet, export_wallet
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_wallet_operations():
    try:
        wallet_data = {
            "user_id": 1,
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "balance": 100.0
        }
        import_result = await import_wallet(wallet_data)
        assert import_result["status"] == "success"
        export_result = await export_wallet(1)
        assert export_result[0]["user_id"] == 1
    except Exception as e:
        error_logger.log_error("test_wallet", str(e), str(e.__traceback__))
        logger.error(f"Wallet test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #wallet #neon_mcp
