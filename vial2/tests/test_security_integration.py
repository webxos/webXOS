import pytest
from ..security.security_tester import security_tester
import logging
import os

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_security_injection():
    try:
        os.environ["AUTH_API_KEY"] = "test_key"
        assert not security_tester.test_injection("normal_input")
        with pytest.raises(ValueError):
            security_tester.test_injection("SELECT * FROM users; <script>alert('xss')</script>")
        logger.info("Security integration test passed")
    except Exception as e:
        logger.error(f"Security integration test failed: {str(e)}")
        raise
    finally:
        if "AUTH_API_KEY" in os.environ:
            del os.environ["AUTH_API_KEY"]

# xAI Artifact Tags: #vial2 #tests #mcp #security #integration #langchain #neon_mcp
