import pytest
from mcp.monitoring.health_monitor import check_health
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_performance():
    try:
        start_time = time.time()
        result = await check_health()
        end_time = time.time()
        assert end_time - start_time < 0.1
        assert result["status"] == "healthy"
        logger.info("Performance test passed")
    except Exception as e:
        logger.error(f"Performance test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #performance #neon_mcp
