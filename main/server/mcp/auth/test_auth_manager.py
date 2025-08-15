# main/server/mcp/tests/test_auth_manager.py
import pytest
from ..auth.auth_manager import AuthManager, MCPError
from ..utils.cache_manager import CacheManager
import asyncio

@pytest.fixture
async def auth_manager():
    manager = AuthManager({"address": "0x123", "hash": "abc123", "reputation": 1000})
    yield manager
    await manager.cache.close()

@pytest.mark.asyncio
async def test_authenticate_success(auth_manager, mocker):
    mocker.patch.object(auth_manager.cache, "set_cache", return_value=None)
    result = await auth_manager.authenticate("test_user", "test_pass")
    assert "access_token" in result
    assert "redirect" in result
    assert result["redirect"] == "/dashboard"

@pytest.mark.asyncio
async def test_authenticate_invalid_credentials(auth_manager):
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.authenticate("test_user", "wrong_pass")
    assert exc_info.value.code == -32001
    assert "Invalid credentials" in exc_info.value.message

@pytest.mark.asyncio
async def test_authenticate_missing_params(auth_manager):
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.authenticate("", "test_pass")
    assert exc_info.value.code == -32602
    assert "Username and password are required" in exc_info.value.message

@pytest.mark.asyncio
async def test_verify_token_valid(auth_manager, mocker):
    mocker.patch.object(auth_manager.cache, "get_cache", return_value={"token": "valid_token"})
    result = await auth_manager.verify_token("valid_token")
    assert result is True

@pytest.mark.asyncio
async def test_verify_token_expired(auth_manager, mocker):
    mocker.patch("jwt.decode", side_effect=jwt.ExpiredSignatureError)
    with pytest.raises(MCPError) as exc_info:
        await auth_manager.verify_token("expired_token")
    assert exc_info.value.code == -32002
    assert "Token has expired" in exc_info.value.message

@pytest.mark.asyncio
async def test_logout_success(auth_manager, mocker):
    mocker.patch.object(auth_manager.cache, "delete_cache", return_value=None)
    result = await auth_manager.logout("test_user")
    assert result is True
