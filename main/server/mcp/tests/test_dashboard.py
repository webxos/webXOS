# main/server/mcp/tests/test_dashboard.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_dashboard_metrics(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    token = response.json()["access_token"]
    status_response = client.post(
        "/mcp/status",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.getSystemMetrics", "params": {"user_id": "test_user"}, "id": 1}
    )
    assert status_response.status_code == 200
    data = status_response.json()
    assert "result" in data
    assert "cpu_usage" in data["result"]