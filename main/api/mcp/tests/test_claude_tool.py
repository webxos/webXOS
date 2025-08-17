import pytest
from fastapi.testclient import TestClient
from main import app, MCPServer
from tools.claude_tool import ClaudeTool
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
async def test_claude_tool_success(client, mock_db):
    mock_db.query.side_effect = [
        type("Result", (), {"rows": [{"user_id": "user_12345"}] }),  # User exists
        type("Result", (), {"rows": [{}]} )  # Code execution stored
    ]
    
    with patch("tools.claude_tool.CodeValidator.is_safe_code", return_value=True):
        server = MCPServer()
        server.tools["claude"] = ClaudeTool(mock_db)
        
        response = client.post("/mcp/execute", json={
            "jsonrpc": "2.0",
            "method": "claude.executeCode",
            "params": {
                "code": "print('Hello, Claude!')",
                "user_id": "user_12345"
            },
            "id": 1
        })
        
        assert response.status_code == 200
        assert response.json()["result"]["output"] == "Hello, Claude!\n"
        assert response.json()["result"]["error"] is None

@pytest.mark.asyncio
async def test_claude_tool_unsafe_code(client, mock_db):
    mock_db.query.return_value = type("Result", (), {"rows": [{"user_id": "user_12345"}]})
    
    with patch("tools.claude_tool.CodeValidator.is_safe_code", return_value=False):
        server = MCPServer()
        server.tools["claude"] = ClaudeTool(mock_db)
        
        response = client.post("/mcp/execute", json={
            "jsonrpc": "2.0",
            "method": "claude.executeCode",
            "params": {
                "code": "import os; os.system('rm -rf /')",
                "user_id": "user_12345"
            },
            "id": 1
        })
        
        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Unsafe code detected"
