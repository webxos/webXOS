# main/server/mcp/tests/test_auth_sync.py
import pytest
from ..sync.auth_sync import AuthSync  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def auth_sync():
    return AuthSync()

@pytest.mark.asyncio
async def test_sync_auth(auth_sync, mocker):
    mocker.patch.object(auth_sync, "sync", return_value=True)
    result = await auth_sync.sync("test_user", "token")
    assert result is True

@pytest.mark.asyncio
async def test_sync_auth_error(auth_sync, mocker):
    mocker.patch.object(auth_sync, "sync", side_effect=MCPError(code=-32603, message="Sync failed"))
    with pytest.raises(MCPError) as exc_info:
        await auth_sync.sync("test_user", "invalid_token")
    assert exc_info.value.code == -32603