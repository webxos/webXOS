# main/server/mcp/auth/test_auth_sync.py
import pytest
from pymongo import MongoClient
from ..auth.auth_sync import AuthSync, MCPError
from datetime import datetime, timedelta

@pytest.fixture
async def auth_sync():
    sync = AuthSync()
    yield sync
    sync.sessions.delete_many({})
    sync.mfa_tokens.delete_many({})
    sync.close()

@pytest.mark.asyncio
async def test_create_session(auth_sync):
    result = await auth_sync.create_session("test_user")
    assert "session_id" in result
    assert "access_token" in result
    assert "expires_at" in result
    session = sync.sessions.find_one({"session_id": result["session_id"]})
    assert session["user_id"] == "test_user"
    assert session["mfa_verified"] is False

@pytest.mark.asyncio
async def test_create_session_invalid(auth_sync):
    with pytest.raises(MCPError) as exc_info:
        await auth_sync.create_session("")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "User ID is required"

@pytest.mark.asyncio
async def test_initiate_mfa(auth_sync):
    result = await auth_sync.initiate_mfa("test_user", "email")
    assert "mfa_token" in result
    assert result["method"] == "email"
    assert "expires_at" in result
    mfa_record = sync.mfa_tokens.find_one({"mfa_token": result["mfa_token"]})
    assert mfa_record["user_id"] == "test_user"

@pytest.mark.asyncio
async def test_initiate_mfa_invalid_method(auth_sync):
    with pytest.raises(MCPError) as exc_info:
        await auth_sync.initiate_mfa("test_user", "invalid")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Unsupported MFA method"

@pytest.mark.asyncio
async def test_verify_mfa(auth_sync, mocker):
    mocker.patch.object(auth_sync.secrets_manager, 'retrieve_secret', return_value="test_mfa_token")
    await auth_sync.create_session("test_user")
    mfa_result = await auth_sync.initiate_mfa("test_user", "email")
    result = await auth_sync.verify_mfa("test_user", mfa_result["mfa_token"], "test_mfa_token")
    assert result["status"] == "success"
    session = sync.sessions.find_one({"user_id": "test_user"})
    assert session["mfa_verified"] is True

@pytest.mark.asyncio
async def test_revoke_session(auth_sync):
    session = await auth_sync.create_session("test_user")
    await auth_sync.revoke_session(session["session_id"], "test_user")
    assert sync.sessions.find_one({"session_id": session["session_id"]}) is None
