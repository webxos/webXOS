# main/server/mcp/tests/test_chatbot.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_chatbot_ping(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    token = response.json()["access_token"]
    ping_response = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.ping", "params": {"user_id": "test_user"}, "id": 1}
    )
    assert ping_response.status_code == 200
    data = ping_response.json()
    assert data["result"]["status"] == "pong"