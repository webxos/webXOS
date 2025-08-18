import pytest
from ..utils.rate_monitor import monitor_api_rates
from ..utils.helpers import get_db_pool
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_monitor_api_rates():
    try:
        async with get_db_pool() as db:
            endpoint = "/mcp/api/status"
            client_ip = "127.0.0.1"
            result = await monitor_api_rates(endpoint, client_ip)
            assert result["status"] == "success"
            assert result["endpoint"] == endpoint
            assert result["client_ip"] == client_ip
            assert result["request_count"] >= 1
    except Exception as e:
        error_logger.log_error("test_rate_monitor", f"Test monitor_api_rates failed: {str(e)}", str(e.__traceback__))
        raise

# xAI Artifact Tags: #vial2 #tests #rate_monitor #neon_mcp
