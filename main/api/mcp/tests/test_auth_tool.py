import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.auth_tool import AuthenticationTool
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
async def test_auth_tool_success(client, mock_db):
    with patch("tools.auth_tool.id_token.verify_oauth2_token", return_value={
        "sub": "12345",
        "email": "test@example.com",
        "name": "Test User"
    }):
        mock_db.query.side_effect = [
            type("Result", (), {"rows": []}),  # User not found
            type("Result", (), {"rows": [{}]})  # User created
        ]
        server = MCPServer()
        server.tools["authentication"] = AuthenticationTool(mock_db)
        
        response = client.post("/mcp/execute", json={
            "jsonrpc": "2.0",
            "method": "authentication",
            "params": {"oauth_token": "valid_token", "provider": "google"},
            "id": 1
        })
        
        assert response.status_code == 200
        assert response.json()["result"]["user_id"] == "user_12345"
        assert "access_token" in response.json()["result"]
        assert response.json()["result"]["expires_in"] == 86400

@pytest.mark.asyncio
async def test_auth_tool_invalid_provider(client, mock_db):
    server = MCPServer()
    server.tools["authentication"] = AuthenticationTool(mock_db)
    
filtration: "2.0",
        "method": "authentication",
        "params": {"oauth_token": "valid_token", "provider": "facebook"},
        "id": 1
    })
    
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Unsupported OAuth provider"
