import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.blockchain import BlockchainTool
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
async def test_blockchain_get_info_success(client, mock_db):
    mock_db.query.return_value = type("Result", (), {
        "rows": [{"count": 100, "last_hash": "abc123"}]
    })
    
    server = MCPServer()
    server.tools["blockchain"] = BlockchainTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "blockchain.getBlockchainInfo",
        "params": {},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["block_count"] == 100
    assert result["last_hash"] == "abc123"

@pytest.mark.asyncio
async def test_blockchain_no_data(client, mock_db):
    mock_db.query.return_value = type("Result", (), {"rows": []})
    
    server = MCPServer()
    server.tools["blockchain"] = BlockchainTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "blockchain.getBlockchainInfo",
        "params": {},
        "id": 1
    })
    
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "No blockchain data found"
