# main/server/mcp/auth/test_mcp_server_auth.py
import pytest
from .mcp_server_auth import authenticate_user, authenticate_oauth

@pytest.mark.asyncio
async def test_authenticate_user():
    token, vials = await authenticate_user("test_user", "test_pass")
    assert token
    assert len(vials) == 4

@pytest.mark.asyncio
async def test_authenticate_user_invalid():
    with pytest.raises(Exception):
        await authenticate_user("wrong_user", "wrong_pass")

@pytest.mark.asyncio
async def test_authenticate_oauth():
    response = await authenticate_oauth("mock", "test_code")
    assert "access_token" in response
    assert "vials" in response
