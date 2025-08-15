# main/server/mcp/utils/test_performance_metrics.py
import pytest
from ..utils.performance_metrics import PerformanceMetrics, MCPError

@pytest.fixture
def metrics():
    return PerformanceMetrics()

@pytest.mark.asyncio
async def test_record_request(metrics):
    metrics.record_request("/mcp", 0.123)
    metrics_data = metrics.get_metrics()
    assert metrics_data["requests_total"][("/mcp",)] == 1
    assert metrics_data["request_latency"][("/mcp",)] == 0.123

@pytest.mark.asyncio
async def test_record_error(metrics):
    metrics.record_error("/mcp", -32601)
    metrics_data = metrics.get_metrics()
    assert metrics_data["errors_total"][("/mcp", "-32601")] == 1

@pytest.mark.asyncio
async def test_set_active_agents(metrics):
    metrics.set_active_agents("vial1", 5)
    metrics_data = metrics.get_metrics()
    assert metrics_data["active_agents"][("vial1",)] == 5

@pytest.mark.asyncio
async def test_get_metrics(metrics):
    metrics.record_request("/mcp", 0.123)
    metrics.record_error("/mcp", -32601)
    metrics.set_active_agents("vial1", 5)
    metrics_data = metrics.get_metrics()
    assert "requests_total" in metrics_data
    assert "request_latency" in metrics_data
    assert "errors_total" in metrics_data
    assert "active_agents" in metrics_data
