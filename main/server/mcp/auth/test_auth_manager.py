# main/server/mcp/auth/test_auth_manager.py
import pytest
from ..auth.auth_manager import authenticate_user

@pytest.mark.asyncio
async def test_authenticate_user():
    token, vials = await authenticate_user("test_user", "test_pass")
    assert token
    assert isinstance(vials, dict)
    assert len(vials) == 4

@pytest.mark.asyncio
async def test_authenticate_user_invalid():
    with pytest.raises(Exception):
        await authenticate_user("wrong_user", "wrong_pass")
