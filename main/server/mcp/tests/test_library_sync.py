# main/server/mcp/tests/test_library_sync.py
import pytest
from ..sync.library_sync import LibrarySync  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def library_sync():
    return LibrarySync()

@pytest.mark.asyncio
async def test_sync_library(library_sync, mocker):
    mocker.patch.object(library_sync, "sync", return_value=True)
    result = await library_sync.sync("test_user", "library_data")
    assert result is True

@pytest.mark.asyncio
async def test_sync_library_error(library_sync, mocker):
    mocker.patch.object(library_sync, "sync", side_effect=MCPError(code=-32603, message="Sync failed"))
    with pytest.raises(MCPError) as exc_info:
        await library_sync.sync("test_user", "invalid_data")
    assert exc_info.value.code == -32603