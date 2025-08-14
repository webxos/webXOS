import pytest,asyncio
from fastapi.testclient import TestClient
from main.server.unified_server import app
from main.server.mcp.mcp_server_resources import MCPResourcesHandler
from main.server.mcp.mcp_auth_server import MCPAuthServer

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def resources_handler():
    """Create an MCPResourcesHandler instance."""
    return MCPResourcesHandler()

@pytest.mark.asyncio
async def test_get_resources_success(client,resources_handler,mocker):
    """Test successful resource retrieval with valid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=True)
    mocker.patch("sqlite3.connect",return_value=mocker.MagicMock(execute=mocker.MagicMock(return_value=[
        (1,"Test note","res_001","2025-08-13T21:10:00Z")
    ])))
    response=client.post("/api/resources/latest",json={"wallet_id":"wallet_123","limit":10},headers={"Authorization":"Bearer test_token"})
    assert response.status_code==200
    assert response.json()["status"]=="success"
    assert len(response.json()["resources"])==1
    assert response.json()["resources"][0]["content"]=="Test note"

@pytest.mark.asyncio
async def test_get_resources_invalid_token(client,resources_handler,mocker):
    """Test resource retrieval with invalid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=False)
    response=client.post("/api/resources/latest",json={"wallet_id":"wallet_123","limit":10},headers={"Authorization":"Bearer invalid_token"})
    assert response.status_code==401
    assert response.json()["detail"]=="Invalid access token"