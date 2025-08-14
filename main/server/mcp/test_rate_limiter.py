import pytest
from fastapi.testclient import TestClient
from main.server.mcp.rate_limiter import RateLimiter
from main.server.unified_server import app
import redis

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def rate_limiter():
    """Create a RateLimiter instance with low limits for testing."""
    return RateLimiter(limit=2, window=60)

@pytest.mark.asyncio
async def test_rate_limit_within_limit(rate_limiter, mocker):
    """Test rate limiting within allowed limit."""
    mocker.patch.object(redis.Redis, 'get', side_effect=[None, "1"])
    mocker.patch.object(redis.Redis, 'setex', return_value=True)
    mocker.patch.object(redis.Redis, 'incr', return_value=2)
    assert rate_limiter.check_limit("wallet_123", "/api/notes/add") is True
    redis.Redis.setex.assert_called_once()
    redis.Redis.incr.assert_called_once()

@pytest.mark.asyncio
async def test_rate_limit_exceeded(rate_limiter, mocker):
    """Test rate limiting when limit is exceeded."""
    mocker.patch.object(redis.Redis, 'get', return_value="2")
    with pytest.raises(HTTPException) as exc:
        rate_limiter.check_limit("wallet_123", "/api/notes/add")
    assert exc.value.status_code == 429
    assert exc.value.detail == "Rate limit exceeded"

@pytest.mark.asyncio
async def test_rate_limit_new_key(rate_limiter, mocker):
    """Test rate limiting for a new key."""
    mocker.patch.object(redis.Redis, 'get', return_value=None)
    mocker.patch.object(redis.Redis, 'setex', return_value=True)
    assert rate_limiter.check_limit("wallet_123", "/api/notes/add") is True
    redis.Redis.setex.assert_called_with("rate:wallet_123:/api/notes/add", 60, 1)
