import pytest,asyncio
from fastapi.testclient import TestClient
from main.server.unified_server import app
from main.server.mcp.mcp_server_notes import MCPNotesHandler
from main.server.mcp.mcp_auth_server import MCPAuthServer

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def notes_handler():
    """Create an MCPNotesHandler instance."""
    return MCPNotesHandler()

@pytest.mark.asyncio
async def test_add_note_success(client,notes_handler,mocker):
    """Test successful note addition with valid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=True)
    mocker.patch("sqlite3.connect",autospec=True)
    response=client.post("/api/notes/add",json={"wallet_id":"wallet_123","content":"Test note"},headers={"Authorization":"Bearer test_token"})
    assert response.status_code==200
    assert response.json()["status"]=="success"
    assert "note_id" in response.json()

@pytest.mark.asyncio
async def test_add_note_invalid_token(client,notes_handler,mocker):
    """Test note addition with invalid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=False)
    response=client.post("/api/notes/add",json={"wallet_id":"wallet_123","content":"Test note"},headers={"Authorization":"Bearer invalid_token"})
    assert response.status_code==401
    assert response.json()["detail"]=="Invalid access token"

@pytest.mark.asyncio
async def test_read_note_success(client,notes_handler,mocker):
    """Test successful note reading with valid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=True)
    mocker.patch("sqlite3.connect",return_value=mocker.MagicMock(execute=mocker.MagicMock(return_value=[(1,"Test note","res_001","2025-08-13T21:10:00Z")]))
    response=client.post("/api/notes/read",json={"note_id":1,"wallet_id":"wallet_123"},headers={"Authorization":"Bearer test_token"})
    assert response.status_code==200
    assert response.json()["status"]=="success"
    assert response.json()["note"]["content"]=="Test note"