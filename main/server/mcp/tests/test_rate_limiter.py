# main/server/mcp/tests/test_rate_limiter.py
import pytest
from ..utils.rate_limiter import RateLimiter  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def rate_limiter():
    return RateLimiter(limit=5, window=60)

@pytest.mark.asyncio
async def test_rate_limit_success(rate_limiter):
    for _ in range(3):
        await rate_limiter.check("test_user")
    assert True

@pytest.mark.asyncio
async def test_rate_limit_exceeded(rate_limiter):
    for _ in range(6):
        await rate_limiter.check("test_user")
    with pytest.raises(MCPError) as exc_info:
        await rate_limiter.check("test_user")
    assert exc_info.value.code == -32604