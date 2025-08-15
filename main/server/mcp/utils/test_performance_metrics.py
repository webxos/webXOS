# main/server/mcp/utils/test_performance_metrics.py
import pytest
from .performance_metrics import get_metrics

@pytest.mark.asyncio
async def test_get_metrics():
    metrics = await get_metrics()
    assert "cpu_usage" in metrics
    assert "memory_usage" in metrics
    assert "active_users" in metrics
