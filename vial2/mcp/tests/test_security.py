import pytest
from mcp.security.security_tester import security_tester
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_security():
    try:
        result = await security_tester.test_security()
        assert result["status"] == "secure"
        logger.info("Security test passed")
    except Exception as e:
        logger.error(f"Security test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #security #neon_mcp
