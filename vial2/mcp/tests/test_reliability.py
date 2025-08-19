import pytest
from mcp.monitoring.resource_alert import check_resource_alert
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_reliability():
    try:
        result = await check_resource_alert()
        assert result["status"] in ["healthy", "unhealthy"]
        logger.info("Reliability test passed")
    except Exception as e:
        logger.error(f"Reliability test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #reliability #neon_mcp
