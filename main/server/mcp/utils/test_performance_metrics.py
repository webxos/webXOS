# main/server/mcp/utils/test_performance_metrics.py
import pytest
from ..utils.performance_metrics import PerformanceMetrics, MCPError
import time

@pytest.fixture
async def metrics():
    return PerformanceMetrics()

@pytest.mark.asyncio
async def test_track_request_success(metrics):
    @metrics.track_request("test_endpoint")
    async def test_func():
        return {"status": "success"}
    
    result = await test_func()
    assert result == {"status": "success"}
    assert metrics.requests_total.labels(endpoint="test_endpoint")._value.get() == 1
    assert metrics.request_duration.labels(endpoint="test_endpoint")._value.get() > 0

@pytest.mark.asyncio
async def test_track_request_error(metrics):
    @metrics.track_request("error_endpoint")
    async def test_func():
        raise MCPError(code=-32602, message="Test error")
    
    with pytest.raises(MCPError) as exc_info:
        await test_func()
    assert exc_info.value.code == -32602
    assert metrics.error_rate.labels(endpoint="error_endpoint", error_code=-32602)._value.get() == 1

@pytest.mark.asyncio
async def test_update_active_users(metrics):
    metrics.update_active_users(10)
    assert metrics.active_users._value.get() == 10

@pytest.mark.asyncio
async def test_get_metrics(metrics):
    @metrics.track_request("test_endpoint")
    async def test_func():
        return {"status": "success"}
    
    await test_func()
    metrics.update_active_users(5)
    result = await metrics.get_metrics()
    assert "requests_total" in result
    assert "request_duration" in result
    assert "sla_compliance" in result
    assert result["active_users"] == 5
    assert result["uptime_hours"] > 0
