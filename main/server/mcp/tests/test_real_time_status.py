# main/server/mcp/tests/test_real_time_status.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json
import time

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_real_time_status(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    token = response.json()["access_token"]
    first_response = client.post(
        "/mcp/status",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.getSystemMetrics", "params": {"user_id": "test_user"}, "id": 1}
    )
    time.sleep(6)  # Allow for update interval
    second_response = client.post(
        "/mcp/status",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.getSystemMetrics", "params": { "user_id": "test_user"}, "id": 1}
    )
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_data = first_response.json()["result"]["balance"]
    second_data = second_response.json()["result"]["balance"]
    assert first_data != second_data or first_response.json() != second_response.json()  # Expect fluctuation