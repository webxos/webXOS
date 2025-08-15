# main/server/mcp/tests/test_import_export.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_import_export_cycle(client):
    response = client.post("/mcp/auth", json={"username": "test_user", "password": "test_pass"})
    token = response.json()["access_token"]
    import_response = client.post(
        "/mcp/import",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.importMd", "params": {"user_id": "test_user", "md_content": "# Test Data"}, "id": 4}
    )
    assert import_response.status_code == 200
    export_response = client.post(
        "/mcp/export",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "method": "mcp.exportMd", "params": {"user_id": "test_user"}, "id": 3}
    )
    assert export_response.status_code == 200
    data = export_response.json()
    assert "# Vial Data" in data["result"]