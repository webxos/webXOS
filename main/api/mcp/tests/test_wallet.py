import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.wallet import WalletTool
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
async def test_wallet_get_vial_balance_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345"}] }),  # User exists
        type("Result", (), {"rows": [{"balance": 100.0}] })  # Balance data
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.getVialBalance",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["vial_id"] == "vial1"
    assert result["balance"] == 100.0

@pytest.mark.asyncio
async def test_wallet_user_not_found(client, mock_db):
    mock_db.query.return_value = type("Result", (), {"rows": []})
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.getVialBalance",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "User not found: user_12345"
