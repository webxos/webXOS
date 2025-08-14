import pytest
from fastapi.testclient import TestClient
from main.server.mcp.api_gateway.gateway_router import GatewayRouter
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app
from datetime import datetime

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def gateway_router():
    """Create a GatewayRouter instance."""
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return GatewayRouter(security_manager, error_handler)

async def mock_handler(payload: dict, access_token: str) -> dict:
    """Mock handler for testing."""
    return {"response": "success", "payload": payload}

def test_register_route(gateway_router):
    """Test registering a route."""
    gateway_router.register_route("/test", mock_handler)
    assert "/test" in gateway_router.routes
    assert gateway_router.routes["/test"] == mock_handler

def test_register_duplicate_route(gateway_router):
    """Test registering a duplicate route."""
    gateway_router.register_route("/test", mock_handler)
    with pytest.raises(HTTPException) as exc:
        gateway_router.register_route("/test", mock_handler)
    assert exc.value.status_code == 500
    assert "Endpoint /test already registered" in exc.value.detail

@pytest.mark.asyncio
async def test_route_request_success(gateway_router, mocker):
    """Test routing a request successfully."""
    mocker.patch.object(gateway_router.security_manager, "validate_token", return_value={"wallet_id": "wallet_123"})
    gateway_router.register_route("/test", mock_handler)
    response = await gateway_router.route_request("/test", {"data": "test"}, "valid_token")
    assert response == {"response": "success", "payload": {"data": "test"}}

@pytest.mark.asyncio
async def test_route_request_not_found(gateway_router):
    """Test routing to a non-existent endpoint."""
    with pytest.raises(HTTPException) as exc:
        await gateway_router.route_request("/invalid", {"data": "test"}, "valid_token")
    assert exc.value.status_code == 500
    assert "Endpoint /invalid not found" in exc.value.detail
