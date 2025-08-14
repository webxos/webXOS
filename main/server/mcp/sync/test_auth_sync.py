import pytest
from fastapi.testclient import TestClient
from main.server.mcp.sync.auth_sync import AuthSyncManager, AuthSyncRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def auth_sync_manager():
    """Create an AuthSyncManager instance."""
    db_manager = DatabaseManager()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return AuthSyncManager(db_manager, security_manager, error_handler)

@pytest.mark.asyncio
async def test_sync_auth_token_success(auth_sync_manager, mocker):
    """Test successful token synchronization."""
    mocker.patch.object(auth_sync_manager.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(auth_sync_manager.db_manager, 'update_token', return_value=None)
    request = AuthSyncRequest(wallet_id="wallet_123", access_token="mocked_token")
    response = await auth_sync_manager.sync_auth_token(request)
    assert response == {"status": "success", "wallet_id": "wallet_123"}

@pytest.mark.asyncio
async def test_sync_auth_token_invalid(auth_sync_manager, mocker):
    """Test token synchronization with invalid token."""
    mocker.patch.object(auth_sync_manager.security_manager, 'validate_token', return_value={"wallet_id": "wrong_wallet"})
    request = AuthSyncRequest(wallet_id="wallet_123", access_token="mocked_token")
    with pytest.raises(HTTPException) as exc:
        await auth_sync_manager.sync_auth_token(request)
    assert exc.value.status_code == 500
    assert "Invalid token for wallet" in exc.value.detail
