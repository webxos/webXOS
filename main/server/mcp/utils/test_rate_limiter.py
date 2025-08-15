# main/server/mcp/utils/test_rate_limiter.py
import pytest
from ..utils.rate_limiter import RateLimiter
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def rate_limiter():
    return RateLimiter(limit=2, window=1)

@pytest.mark.asyncio
async def test_check_rate_limiter(rate_limiter):
    assert await rate_limiter.check("user1") is True
    assert await rate_limiter.check("user1") is True
    with pytest.raises(MCPError):
        await rate_limiter.check("user1")
