import pytest
from fastapi.testclient import TestClient
from main.server.mcp.auth_sync import AuthSync
from main.server.mcp.auth_manager import AuthManager
from main.server.unified_server import app
import redis

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def auth_sync():
    """Create an AuthSync instance."""
    return AuthSync()

@pytest.mark.asyncio
async def test_sync_token_success(auth_sync, mocker):
    """Test successful token synchronization."""
    mocker.patch.object(redis.Redis, 'setex', return_value=True)
    auth_sync.sync_token("wallet_123", "test_token", ttl=3600)
    redis.Redis.setex.assert_called_with("auth:wallet_123", 3600, "test_token")

@pytest.mark.asyncio
async def test_get_synced_token_hit(auth_sync, mocker):
    """Test retrieving a synced token."""
    mocker.patch.object(redis.Redis, 'get', return_value="test_token")
    result = auth_sync.get_synced_token("wallet_123")
    assert result == "test_token"
    redis.Redis.get.assert_called_with("auth:wallet_123")

@pytest.mark.asyncio
async def test_get_synced_token_miss(auth_sync, mocker):
    """Test retrieving a non-existent synced token."""
    mocker.patch.object(redis.Redis, 'get', return_value=None)
    result = auth_sync.get_synced_token("wallet_123")
    assert result is None
    redis.Redis.get.assert_called_with("auth:wallet_123")

@pytest.mark.asyncio
async def test_verify_synced_token_success(auth_sync, mocker):
    """Test verifying a valid synced token."""
    mocker.patch.object(redis.Redis, 'get', return_value="test_token")
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_123"})
    result = auth_sync.verify_synced_token("wallet_123", "test_token")
    assert result is True
    redis.Redis.get.assert_called_with("auth:wallet_123")
    AuthManager.verify_token.assert_called_with("test_token")

@pytest.mark.asyncio
async def test_verify_synced_token_invalid(auth_sync, mocker):
    """Test verifying an invalid synced token."""
    mocker.patch.object(redis.Redis, 'get', return_value="different_token")
    result = auth_sync.verify_synced_token("wallet_123", "test_token")
    assert result is False
    redis.Redis.get.assert_called_with("auth:wallet_123")
