import pytest
from fastapi.testclient import TestClient
from main.server.mcp.wallet.webxos_wallet import MCPWalletManager, WalletRequest, WalletUpdateRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def wallet_manager():
    """Create an MCPWalletManager instance."""
    db_manager = DatabaseManager()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return MCPWalletManager(db_manager, security_manager, error_handler)

@pytest.mark.asyncio
async def test_create_wallet_success(wallet_manager, mocker):
    """Test successful wallet creation."""
    mocker.patch.object(wallet_manager.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(wallet_manager.db_manager, 'create_wallet', return_value={"wallet_id": "wallet_123", "user_id": "user_123"})
    request = WalletRequest(user_id="user_123", wallet_id="wallet_123")
    response = await wallet_manager.create_wallet(request, "mocked_token")
    assert response == {"wallet_id": "wallet_123", "user_id": "user_123"}

@pytest.mark.asyncio
async def test_create_wallet_unauthorized(wallet_manager, mocker):
    """Test wallet creation with unauthorized wallet."""
    mocker.patch.object(wallet_manager.security_manager, 'validate_token', return_value={"wallet_id": "wrong_wallet"})
    request = WalletRequest(user_id="user_123", wallet_id="wallet_123")
    with pytest.raises(HTTPException) as exc:
        await wallet_manager.create_wallet(request, "mocked_token")
    assert exc.value.status_code == 500
    assert "Unauthorized wallet access" in exc.value.detail

@pytest.mark.asyncio
async def test_update_wallet_success(wallet_manager, mocker):
    """Test successful wallet update."""
    mocker.patch.object(wallet_manager.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(wallet_manager.db_manager, 'update_wallet', return_value={"wallet_id": "wallet_123", "settings": {"theme": "dark"}})
    request = WalletUpdateRequest(wallet_id="wallet_123", settings={"theme": "dark"})
    response = await wallet_manager.update_wallet(request, "mocked_token")
    assert response == {"wallet_id": "wallet_123", "settings": {"theme": "dark"}}

@pytest.mark.asyncio
async def test_update_wallet_unauthorized(wallet_manager, mocker):
    """Test wallet update with unauthorized wallet."""
    mocker.patch.object(wallet_manager.security_manager, 'validate_token', return_value={"wallet_id": "wrong_wallet"})
    request = WalletUpdateRequest(wallet_id="wallet_123", settings={"theme": "dark"})
    with pytest.raises(HTTPException) as exc:
        await wallet_manager.update_wallet(request, "mocked_token")
    assert exc.value.status_code == 500
    assert "Unauthorized wallet access" in exc.value.detail
