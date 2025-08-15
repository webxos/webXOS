# main/server/mcp/utils/test_rate_limiter.py
import pytest
import redis.asyncio as redis
from ..utils.rate_limiter import RateLimiter, MCPError
import time

@pytest.fixture
async def rate_limiter():
    limiter = RateLimiter()
    yield limiter
    await limiter.clear_rate_limit("test_user")
    await limiter.close()

@pytest.mark.asyncio
async def test_rate_limit_within_limit(rate_limiter):
    for _ in range(50):  # Within limit of 60
        await rate_limiter.check_rate_limit("test_user", "/mcp")
    assert True  # No exception means success

@pytest.mark.asyncio
async def test_rate_limit_exceeded(rate_limiter):
    for _ in range(60):  # Reach limit
        await rate_limiter.check_rate_limit("test_user", "/mcp")
    
    with pytest.raises(MCPError) as exc_info:
        await rate_limiter.check_rate_limit("test_user", "/mcp")
    assert exc_info.value.code == -32029
    assert exc_info.value.message == "Rate limit exceeded"

@pytest.mark.asyncio
async def test_burst_limit(rate_limiter, mocker):
    mocker.patch("time.time", return_value=1000)
    for _ in range(10):  # Within burst limit
        await rate_limiter.check_rate_limit("test_user", "/mcp")
    
    with pytest.raises(MCPError) as exc_info:
        await rate_limiter.check_rate_limit("test_user", "/mcp")
    assert exc_info.value.code == -32029
    assert exc_info.value.message == "Burst limit exceeded"

@pytest.mark.asyncio
async def test_clear_rate_limit(rate_limiter):
    await rate_limiter.check_rate_limit("test_user", "/mcp")
    await rate_limiter.clear_rate_limit("test_user")
    await rate_limiter.check_rate_limit("test_user", "/mcp")  # Should not raise
    assert True
