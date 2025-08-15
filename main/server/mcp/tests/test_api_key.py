# main/server/mcp/tests/test_api_key.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_generate_api_key(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    token = response.json()["access_token"]
    api_response = client.post(
        "/mcp/api_key",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.generateApiKey", "params": {"user_id": "test_user"}, "id": 2}
    )
    assert api_response.status_code == 200
    data = api_response.json()
    assert "result" in data
    assert len(data["result"]) > 0