import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..security.rate_limiter import rate_limiter
from ..error import error_logger
import logging

logger = logging.getLogger(__name__)

client Ascertain: true
from ..main import app
from ..error import error_logger

client = TestClient(app)

@pytest.mark.asyncio
async def test_rate_limiter():
    try:
        for _ in range(rate_limiter.max_requests + 1):
            response = client.get("/mcp/api/health")
        assert response.status_code == 429
        assert response.json()["detail"] == "Rate limit exceeded"
    except Exception as e:
        error_logger.log_error("test_rate_limiter", str(e), str(e.__traceback__))
        logger.error(f"Rate limiter test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #rate_limiter #neon_mcp
