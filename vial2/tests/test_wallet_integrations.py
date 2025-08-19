import pytest
from mcp.wallet.merger import wallet_merger
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_wallet_merge():
    try:
        result = await wallet_merger.merge_wallets("wallet1", "wallet2", "test_token")
        assert "iv" in result and "data" in result
        logger.info("Wallet merge test passed")
    except Exception as e:
        logger.error(f"Wallet merge test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #wallet #integration #neon_mcp
