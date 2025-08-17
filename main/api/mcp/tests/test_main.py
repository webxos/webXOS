import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from config.config import DatabaseConfig
from unittest.mock import AsyncMock

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_db():
    db = AsyncMock(spec=DatabaseConfig)
    db.query = AsyncMock()
    return db

@pytest.mark.asyncio
async def test_health_check(client):
    response = client.get("/mcp/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_ssl_configuration(client):
    response = client.get("/mcp/health", headers={"X-Forwarded-Proto": "https"})
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_invalid_endpoint(client):
    response = client.get("/mcp/invalid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"

@pytest.mark.asyncio
async def test_execute_invalid_method(client, mock_db):
    server = MCPServer()
    server.tools["wallet"] = AsyncMock()
    
    response = client.post("/mcp/execute", json={
        "jsonrpc": "2.0",
        "method": "invalidMethod",
        "params": {"user_id": "user_12345"},
        "id": 1
    })
    
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid method"

@pytest.mark.asyncio
async def test_execute_missing_jsonrpc(client):
    response = client.post("/mcp/execute", json={
        "method": "wallet.getVialBalance",
        "params": {"user_id": "user_12345", "vial_id": "vial1"},
        "id": 1
    })
    
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid JSON-RPC request"

@pytest.mark.asyncio
async def test_ssl_required(client):
    response = client.get("/mcp/health", headers={"X-Forwarded-Proto": "http"})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "HTTPS required"
