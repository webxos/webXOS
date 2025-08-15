# main/server/mcp/tests/test_landing.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_landing_authenticate_success(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "redirect" in data
    assert data["redirect"] == "/dashboard"

@pytest.mark.asyncio
async def test_landing_troubleshoot(client):
    response = client.post("/mcp/checklist", json={})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "all_files_present" in data["result"]