# main/server/mcp/utils/test_rate_limiter.py
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from ..utils.rate_limiter import RateLimiter, MCPError
from ..utils.performance_metrics import PerformanceMetrics

app = FastAPI()

@pytest.fixture
async def rate_limiter():
    limiter = RateLimiter()
    yield limiter
    await limiter.redis_client.flushdb()
    await limiter.close()

@pytest.fixture
def client(rate_limiter):
    @app.get("/test")
    async def test_endpoint(request: Request):
        await rate_limiter.check_rate_limit(request)
        return {"status": "success"}
    return TestClient(app)

@pytest.mark.asyncio
async def test_rate_limit_check(rate_limiter, client, mocker):
    request = mocker.Mock(client=mocker.Mock(host="127.0.0.1"))
    await rate_limiter.check_rate_limit(request)
    count = await rate_limiter.redis_client.get("rate_limit:127.0.0.1")
    assert int(count) == 1
    assert rate_limiter.metrics.requests_total.labels(endpoint="rate_limit")._value.get() == 1

@pytest.mark.asyncio
async def test_rate_limit_exceeded(rate_limiter, client, mocker):
    request = mocker.Mock(client=mocker.Mock(host="127.0.0.1"))
    for _ in range(rate_limiter.rate_limit):
        await rate_limiter.redis_client.incr("rate_limit:127.0.0.1")
    with pytest.raises(HTTPException) as exc_info:
        await rate_limiter.check_rate_limit(request)
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail

@pytest.mark.asyncio
async def test_reset_rate_limit(rate_limiter):
    await rate_limiter.redis_client.setex("rate_limit:127.0.0.1", 60, 50)
    await rate_limiter.reset_rate_limit("127.0.0.1")
    assert await rate_limiter.redis_client.get("rate_limit:127.0.0.1") is None

@pytest.mark.asyncio
async def test_rate_limit_error(rate_limiter, mocker):
    mocker.patch("redis.asyncio.Redis.get", side_effect=Exception("Redis error"))
    request = mocker.Mock(client=mocker.Mock(host="127.0.0.1"))
    with pytest.raises(MCPError) as exc_info:
        await rate_limiter.check_rate_limit(request)
    assert exc_info.value.code == -32603
    assert "Failed to check rate limit" in exc_info.value.message
