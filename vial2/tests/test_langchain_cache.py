import pytest
from ..langchain.cache_manager import cache_manager
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_cache_manager():
    try:
        cache_manager.set_cache("test_key", "test_value")
        assert cache_manager.get_cache("test_key") == "test_value"
        logger.info("LangChain cache test passed")
    except Exception as e:
        logger.error(f"LangChain cache test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #cache #neon_mcp
