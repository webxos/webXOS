import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.wallet import WalletTool
from config.config import DatabaseConfig
from unittest.mock import AsyncMock, patch
import hashlib

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
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] })
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
async def test_wallet_import_success(client, mock_db):
    markdown = "class VialAgent1:\n    def __init__(self):\n        self.balance = 50.0"
    hash_value = hashlib.sha256(markdown.encode()).hexdigest()
    
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {
            "user_id": "user_12345",
            "markdown": markdown,
            "hash": hash_value
        },
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["imported_vials"] == ["vial1"]
    assert result["total_balance"] == 150.0

@pytest.mark.asyncio
async def test_wallet_import_invalid_hash(client, mock_db):
    markdown = "class VialAgent1:\n    def __init__(self):\n        self.balance = 50.0"
    
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {
            "user_id": "user_12345",
            "markdown": markdown,
            "hash": "invalid_hash"
        },
        "id": 1
    })
    
    assert response.status_code == 400
    assert "Hash mismatch" in response.json()["error"]["message"]

@pytest.mark.asyncio
async def test_wallet_export_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.exportVials",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "# Wallet Export for user_12345" in result["markdown"]
    assert "vial1: 100.0" in result["markdown"]
    assert "hash" in result

@pytest.mark.asyncio
async def test_wallet_mine_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.mineVial",
        "params": {"user_id": "user_12345", "vial_id": "vial1", "nonce": 12345},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "hash" in result
    assert "reward" in result
    assert "balance" in result

@pytest.mark.asyncio
async def test_wallet_void_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Balance updated
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.voidVial",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["vial_id"] == "vial1"
    assert result["status"] == "voided"

@pytest.mark.asyncio
async def test_wallet_troubleshoot_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345", "balance": 100.0, "wallet_address": "wallet_12345"}] })
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.troubleshootVial",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["vial_id"] == "vial1"
    assert result["status"] == "operational"
    assert result["diagnostics"]["balance"] == 100.0

@pytest.mark.asyncio
async def test_wallet_quantum_link_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345"}] }),
        type("Result", (), {"rows": [{}]} )  # Link inserted
    ]
    
    server = MCPServer()
    server.tools["wallet"] = WalletTool(mock_db)
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "wallet.quantumLink",
        "params": {"user_id": "user_12345"},
        "id": 1
    })
    
    assert response.status_code == 200
    result = response.json()["result"]
    assert "link_id" in result
