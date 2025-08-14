import pytest,asyncio
from fastapi.testclient import TestClient
from main.server.unified_server import app
from main.server.mcp.mcp_server_auth import MCPAuthHandler
from main.server.mcp.mcp_auth_server import MCPAuthServer

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def auth_handler():
    """Create an MCPAuthHandler instance."""
    return MCPAuthHandler()

@pytest.mark.asyncio
async def test_authenticate_success(client,auth_handler,mocker):
    """Test successful authentication with valid API key and wallet ID."""
    mocker.patch.object(MCPAuthServer,'generate_oauth_token',return_value={
        "access_token":"test_access_token",
        "refresh_token":"test_refresh_token",
        "expires_at":"2025-08-13T21:10:00Z"
    })
    response=client.post("/api/auth/login",json={"api_key":"api-a24cb96b-96cd-488d-a013-91cb8edbbe68","wallet_id":"wallet_123"})
    assert response.status_code==200
    assert response.json()=={
        "access_token":"test_access_token",
        "refresh_token":"test_refresh_token",
        "expires_at":"2025-08-13T21:10:00Z"
    }

@pytest.mark.asyncio
async def test_authenticate_invalid_api_key(client,auth_handler,mocker):
    """Test authentication with invalid API key."""
    mocker.patch.object(MCPAuthServer,'generate_oauth_token',side_effect=Exception("Invalid API key"))
    response=client.post("/api/auth/login",json={"api_key":"invalid_key","wallet_id":"wallet_123"})
    assert response.status_code==500
    assert "Invalid API key" in response.json()["detail"]

@pytest.mark.asyncio
async def test_refresh_token_success(client,auth_handler,mocker):
    """Test successful token refresh with valid refresh token."""
    mocker.patch.object(MCPAuthServer,'refresh_oauth_token',return_value={
        "access_token":"new_access_token",
        "expires_at":"2025-08-13T21:10:00Z"
    })
    response=client.post("/api/auth/refresh",json={"refresh_token":"test_refresh_token","wallet_id":"wallet_123"})
    assert response.status_code==200
    assert response.json()=={
        "access_token":"new_access_token",
        "expires_at":"2025-08-13T21:10:00Z"
    }