import pytest
import redis
from fastapi.testclient import TestClient
from main.server.mcp.cache_manager import CacheManager
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def cache_manager():
    """Create a CacheManager instance."""
    return CacheManager()

@pytest.mark.asyncio
async def test_cache_response_success(cache_manager, mocker):
    """Test successful caching of a response."""
    mocker.patch.object(redis.Redis, 'setex', return_value=True)
    response = {"status": "success", "data": "test"}
    cache_manager.cache_response("test_key", response, ttl=300)
    redis.Redis.setex.assert_called_with("test_key", 300, mocker.ANY)

@pytest.mark.asyncio
async def test_get_cached_response_hit(cache_manager, mocker):
    """Test retrieving a cached response."""
    mocker.patch.object(redis.Redis, 'get', return_value='{"status":"success","data":"test"}')
    result = cache_manager.get_cached_response("test_key")
    assert result == {"status": "success", "data": "test"}
    redis.Redis.get.assert_called_with("test_key")

@pytest.mark.asyncio
async def test_get_cached_response_miss(cache_manager, mocker):
    """Test cache miss scenario."""
    mocker.patch.object(redis.Redis, 'get', return_value=None)
    result = cache_manager.get_cached_response("test_key")
    assert result is None
    redis.Redis.get.assert_called_with("test_key")

@pytest.mark.asyncio
async def test_clear_cache_success(cache_manager, mocker):
    """Test clearing the cache."""
    mocker.patch.object(redis.Redis, 'scan_iter', return_value=["test_key"])
    mocker.patch.object(redis.Redis, 'delete', return_value=True)
    cache_manager.clear_cache("test_*")
    redis.Redis.scan_iter.assert_called_with("test_*")
    redis.Redis.delete.assert_called_with("test_key")
