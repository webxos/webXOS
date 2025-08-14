import pytest
from fastapi.testclient import TestClient
from main.server.mcp.auth.mcp_auth_server import MCPAuthServer, AuthRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def auth_server():
    """Create an MCPAuthServer instance."""
    db_manager = DatabaseManager()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return MCPAuthServer(db_manager, security_manager, error_handler)

@pytest.mark.asyncio
async def test_authenticate_success(auth_server, mocker):
    """Test successful authentication."""
    mocker.patch.object(auth_server.db_manager, 'get_user', return_value={"wallet_id": "wallet_123", "api_key": "key_123"})
    mocker.patch.object(auth_server.security_manager, 'generate_token', return_value="mocked_token")
    request = AuthRequest(wallet_id="wallet_123", api_key="key_123")
    response = await auth_server.authenticate(request)
    assert response == {"access_token": "mocked_token", "token_type": "bearer"}

@pytest.mark.asyncio
async def test_authenticate_invalid_credentials(auth_server, mocker):
    """Test authentication with invalid credentials."""
    mocker.patch.object(auth_server.db_manager, 'get_user', return_value=None)
    request = AuthRequest(wallet_id="wallet_123", api_key="invalid_key")
    with pytest.raises(HTTPException) as exc:
        await auth_server.authenticate(request)
    assert exc.value.status_code == 500
    assert "Invalid credentials for wallet wallet_123" in exc.value.detail
