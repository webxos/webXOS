import pytest
from fastapi.testclient import TestClient
from main.server.mcp.performance_metrics import PerformanceMetrics
from main.server.unified_server import app
import os
import json

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def performance_metrics():
    """Create a PerformanceMetrics instance."""
    return PerformanceMetrics(metrics_file="/tmp/test_performance_metrics.jsonl")

@pytest.mark.asyncio
async def test_record_endpoint_metrics(performance_metrics, mocker):
    """Test recording endpoint metrics."""
    mocker.patch("builtins.open", mocker.mock_open())
    performance_metrics.record_endpoint_metrics("/api/notes/add", "wallet_123", 0.123, 200)
    open.assert_called_with("/tmp/test_performance_metrics.jsonl", "a")
    open().write.assert_called_once()

@pytest.mark.asyncio
async def test_get_endpoint_metrics_all(performance_metrics, mocker):
    """Test retrieving all performance metrics."""
    mock_metrics = [
        {"endpoint": "/api/notes/add", "wallet_id": "wallet_123", "response_time": 0.123, "status_code": 200},
        {"endpoint": "/api/notes/read", "wallet_id": "wallet_123", "response_time": 0.456, "status_code": 200}
    ]
    mocker.patch("builtins.open", mocker.mock_open(read_data="\n".join([json.dumps(m) for m in mock_metrics])))
    mocker.patch("os.path.exists", return_value=True)
    metrics = performance_metrics.get_endpoint_metrics()
    assert len(metrics) == 2
    assert metrics[0]["endpoint"] == "/api/notes/add"
    assert metrics[1]["endpoint"] == "/api/notes/read"

@pytest.mark.asyncio
async def test_get_endpoint_metrics_filtered(performance_metrics, mocker):
    """Test retrieving filtered performance metrics."""
    mock_metrics = [
        {"endpoint": "/api/notes/add", "wallet_id": "wallet_123", "response_time": 0.123, "status_code": 200},
        {"endpoint": "/api/notes/read", "wallet_id": "wallet_123", "response_time": 0.456, "status_code": 200}
    ]
    mocker.patch("builtins.open", mocker.mock_open(read_data="\n".join([json.dumps(m) for m in mock_metrics])))
    mocker.patch("os.path.exists", return_value=True)
    metrics = performance_metrics.get_endpoint_metrics("/api/notes/add", limit=1)
    assert len(metrics) == 1
    assert metrics[0]["endpoint"] == "/api/notes/add"

@pytest.mark.asyncio
async def test_get_endpoint_metrics_empty(performance_metrics, mocker):
    """Test retrieving metrics when file does not exist."""
    mocker.patch("os.path.exists", return_value=False)
    metrics = performance_metrics.get_endpoint_metrics()
    assert metrics == []
