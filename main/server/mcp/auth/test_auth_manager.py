# main/server/mcp/auth/test_auth_manager.py
import pytest
import base64
from fastapi.testclient import TestClient
from ..auth_manager import AuthManager, AuthCredentials, MCPError

@pytest.fixture
def auth_manager():
    return AuthManager()

@pytest.mark.asyncio
async def test_password_authentication(auth_manager):
    credentials = AuthCredentials(username="test_user", password="secure_password")
    response = await auth_manager.authenticate_user(credentials)
    assert response.token_type == "Bearer"
    assert response.user_id == "test_user"
    assert response.access_token

@pytest.mark.asyncio
async def test_invalid_password(auth_manager):
    credentials = AuthCredentials(username="test_user", password="wrong_password")
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.authenticate_user(credentials)
    assert exc_info.value.code == -32001
    assert exc_info.value.message == "Invalid username or password"

@pytest.mark.asyncio
async def test_wallet_authentication(auth_manager, mocker):
    mocker.patch.object(auth_manager.wallet_service, 'verify_wallet', return_value=True)
    credentials = AuthCredentials(wallet_address="0x1234567890abcdef")
    response = await auth_manager.authenticate_user(credentials)
    assert response.token_type == "Bearer"
    assert response.user_id == "0x1234567890abcdef"
    assert response.access_token

@pytest.mark.asyncio
async def test_invalid_wallet(auth_manager, mocker):
    mocker.patch.object(auth_manager.wallet_service, 'verify_wallet', return_value=False)
    credentials = AuthCredentials(wallet_address="0xinvalid")
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.authenticate_user(credentials)
    assert exc_info.value.code == -32001
    assert exc_info.value.message == "Invalid wallet address"

@pytest.mark.asyncio
async def test_token_validation(auth_manager):
    token = base64.b64encode(b"test_user:1234567890abcdef").decode()
    result = await auth_manager.validate_token(token)
    assert result["user_id"] == "test_user"

@pytest.mark.asyncio
async def test_invalid_token(auth_manager):
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.validate_token("invalid_token")
    assert exc_info.value.code == -32001
    assert exc_info.value.message.startswith("Token validation failed")
