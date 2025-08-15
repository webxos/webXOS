# main/server/mcp/tests/test_vial.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_vial_page_data(client):
    response = client.post(
        "/mcp/auth",
        json={"username": "test_user", "password": "test_pass"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    status_response = client.post(
        "/mcp/status",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.getSystemMetrics", "params": {"user_id": "test_user"}, "id": 1}
    )
    assert status_response.status_code == 200
    data = status_response.json()
    assert "result" in data
    assert "balance" in data["result"]
    assert data["result"]["balance"] > 0

@pytest.mark.asyncio
async def test_export_vial_data(client):
    response = client.post(
        "/mcp/auth",
        json={"username": "test_user", "password": "test_pass"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    export_response = client.post(
        "/mcp/export",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.exportMd", "params": {"user_id": "test_user"}, "id": 3}
    )
    assert export_response.status_code == 200
    data = export_response.json()
    assert "result" in data
    assert "# Vial Data" in data["result"]

@pytest.mark.asyncio
async def test_import_vial_data(client):
    response = client.post(
        "/mcp/auth",
        json={"username": "test_user", "password": "test_pass"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    import_response = client.post(
        "/mcp/import",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.importMd", "params": {"user_id": "test_user", "md_content": "# Test Data"}, "id": 4}
    )
    assert import_response.status_code == 200
    data = import_response.json()
    assert "status" in data
    assert data["status"] == "imported"
