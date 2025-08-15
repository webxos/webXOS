# main/server/mcp/tests/test_library_agent.py
import pytest
from ..agents.library_agent import LibraryAgent  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def library_agent():
    return LibraryAgent()

@pytest.mark.asyncio
async def test_search_library(library_agent, mocker):
    mocker.patch.object(library_agent, "search", return_value=["book1", "book2"])
    result = await library_agent.search("test")
    assert result == ["book1", "book2"]

@pytest.mark.asyncio
async def test_search_library_error(library_agent, mocker):
    mocker.patch.object(library_agent, "search", side_effect=MCPError(code=-32603, message="Search failed"))
    with pytest.raises(MCPError) as exc_info:
        await library_agent.search("invalid")
    assert exc_info.value.code == -32603