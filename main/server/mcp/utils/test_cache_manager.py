# main/server/mcp/utils/test_cache_manager.py
import pytest
import redis.asyncio as redis
from ..utils.cache_manager import CacheManager, MCPError

@pytest.fixture
async def cache_manager():
    manager = CacheManager()
    yield manager
    await manager.clear_cache()
    await manager.close()

@pytest.mark.asyncio
async def test_set_get_cache(cache_manager):
    key = "test_key"
    value = {"data": "test"}
    await cache_manager.set_cache(key, value, ttl=60)
    result = await cache_manager.get_cache(key)
    assert result == value

@pytest.mark.asyncio
async def test_get_nonexistent_cache(cache_manager):
    result = await cache_manager.get_cache("nonexistent_key")
    assert result is None

@pytest.mark.asyncio
async def test_delete_cache(cache_manager):
    key = "test_key"
    value = {"data": "test"}
    await cache_manager.set_cache(key, value)
    await cache_manager.delete_cache(key)
    result = await cache_manager.get_cache(key)
    assert result is None

@pytest.mark.asyncio
async def test_clear_cache(cache_manager):
    await cache_manager.set_cache("key1", {"data": "test1"})
    await cache_manager.set_cache("key2", {"data": "test2"})
    await cache_manager.clear_cache()
    assert await cache_manager.get_cache("key1") is None
    assert await cache_manager.get_cache("key2") is None
