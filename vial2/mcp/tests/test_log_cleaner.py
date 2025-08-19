import pytest
from mcp.maintenance.log_cleaner import log_cleaner
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_log_cleaner():
    try:
        await log_cleaner.clean_old_logs(days=1)
        logger.info("Log cleaner test passed")
    except Exception as e:
        logger.error(f"Log cleaner test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #log #cleaner #neon_mcp
