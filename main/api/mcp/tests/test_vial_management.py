import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.vial_management import VialManagementTool
from config.config import DatabaseConfig
from unittest.mock import AsyncMock, patch

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_db():
    db = AsyncMock(spec=DatabaseConfig)
    db.query = AsyncMock()
    return db

@pytest.mark.asyncio
async def test_vial_management_get_user_data_success(client, mock_db):
    mock_db.query.return_value = type("Result", (), {
        "rows": [{
            "user_id": "user_12345",
            "balance": 100.0,
            "reputation": 1000,
            "wallet_address": "wallet_user_12345"
        }]
    })
    
    server = MCPServer()
    server.tools["vial-management"] = VialManagementTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "vial-management.getUserData",
        "params": {"user_id": "user_12345"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["user_id"] == "user_12345"
    assert result["balance"] == 100.0
    assert result["reputation"] == 1000
    assert result["wallet_address"] == "wallet_user_12345"

@pytest.mark.asyncio
async def test_vial_management_user_not_found(client, mock_db):
    mock_db.query.return_value = type("Result", (), {"rows": []})
    
    server = MCPServer()
    server.tools["vial-management"] = VialManagementTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "vial-management.getUserData",
        "params": {"user_id": "user_12345"},
        "id": 1
    })
    
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "User not found: user_12345"
