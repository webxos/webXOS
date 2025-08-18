import pytest
from ..utils.cache import cache_manager
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_cache_set_get():
    try:
        key = "test_key"
        value = {"data": "test_value"}
        await cache_manager.set_cached_data(key, value)
        result = await cache_manager.get_cached_data(key)
        assert result == value
    except Exception as e:
        error_logger.log_error("test_cache", f"Test cache set/get failed: {str(e)}", str(e.__traceback__))
        raise

@pytest.mark.asyncio
async def test_cache_miss():
    try:
        result = await cache_manager.get_cached_data("nonexistent_key")
        assert result is None
    except Exception as e:
        error_logger.log_error("test_cache", f"Test cache miss failed: {str(e)}", str(e.__traceback__))
        raise

# xAI Artifact Tags: #vial2 #tests #cache #neon_mcp
