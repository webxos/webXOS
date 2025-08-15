# main/server/mcp/utils/test_cache_manager.py
import pytest
from ..utils.cache_manager import CacheManager, MCPError
from ..utils.performance_metrics import PerformanceMetrics

@pytest.fixture
async def cache_manager():
    manager = CacheManager()
    yield manager
    await manager.redis_client.flushdb()
    await manager.close()

@pytest.mark.asyncio
async def test_set_and_get_cache(cache_manager):
    await cache_manager.set_cache("test_key", {"data": "value"}, ttl=60)
    result = await cache_manager.get_cache("test_key")
    assert result == {"data": "value"}
    assert cache_manager.metrics.requests_total.labels(endpoint="set_cache")._value.get() == 1
    assert cache_manager.metrics.requests_total.labels(endpoint="get_cache")._value.get() == 1

@pytest.mark.asyncio
async def test_get_cache_miss(cache_manager):
    result = await cache_manager.get_cache("nonexistent_key")
    assert result is None
    assert cache_manager.metrics.requests_total.labels(endpoint="get_cache")._value.get() == 1

@pytest.mark.asyncio
async def test_delete_cache(cache_manager):
    await cache_manager.set_cache("test_key", {"data": "value"})
    await cache_manager.delete_cache("test_key")
    result = await cache_manager.get_cache("test_key")
    assert result is None
    assert cache_manager.metrics.requests_total.labels(endpoint="delete_cache")._value.get() == 1

@pytest.mark.asyncio
async def test_clear_cache(cache_manager):
    await cache_manager.set_cache("test_key1", {"data": "value1"})
    await cache_manager.set_cache("test_key2", {"data": "value2"})
    await cache_manager.clear_cache("test_key*")
    assert await cache_manager.get_cache("test_key1") is None
    assert await cache_manager.get_cache("test_key2") is None
    assert cache_manager.metrics.requests_total.labels(endpoint="clear_cache")._value.get() == 1

@pytest.mark.asyncio
async def test_invalid_key(cache_manager):
    with pytest.raises(MCPError) as exc_info:
        await cache_manager.set_cache("", {"data": "value"})
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Cache key is required"
