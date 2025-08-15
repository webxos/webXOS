# main/server/mcp/agents/test_library_agent.py
import pytest
from ..agents.library_agent import LibraryAgent, MCPError
from ..utils.performance_metrics import PerformanceMetrics
import aiohttp
import json

@pytest.fixture
async def library_agent():
    agent = LibraryAgent()
    yield agent
    agent.resources.delete_many({})
    agent.close()

@pytest.mark.asyncio
async def test_list_resources(library_agent, mocker):
    resource = {
        "user_id": "test_user",
        "agent_id": "test_agent",
        "type": "github_repo",
        "uri": "https://github.com/test/repo",
        "metadata": {"name": "repo"},
        "created_at": datetime.utcnow()
    }
    library_agent.resources.insert_one(resource)
    resources = await library_agent.list_resources("test_agent", "test_user")
    assert len(resources) == 1
    assert resources[0]["uri"] == "https://github.com/test/repo"
    assert library_agent.metrics.requests_total.labels(endpoint="list_resources")._value.get() == 1

@pytest.mark.asyncio
async def test_list_resources_cached(library_agent, mocker):
    mocker.patch.object(library_agent.cache, "get_cache", return_value=[{"uri": "cached"}])
    resources = await library_agent.list_resources("test_agent", "test_user")
    assert len(resources) == 1
    assert resources[0]["uri"] == "cached"

@pytest.mark.asyncio
async def test_sync_github_resource(library_agent, mocker):
    mocker.patch("aiohttp.ClientSession.get", return_value=mocker.AsyncMock(
        status=200,
        json=mocker.AsyncMock(return_value={
            "html_url": "https://github.com/test/repo",
            "name": "repo",
            "description": "Test repo",
            "updated_at": "2023-01-01T00:00:00Z"
        })
    ))
    result = await library_agent.sync_github_resource("test_user", "test/repo")
    assert "resource_id" in result
    assert result["status"] == "synced"
    resource = library_agent.resources.find_one({"user_id": "test_user"})
    assert resource["uri"] == "https://github.com/test/repo"

@pytest.mark.asyncio
async def test_sync_github_resource_error(library_agent, mocker):
    mocker.patch("aiohttp.ClientSession.get", return_value=mocker.AsyncMock(status=404))
    with pytest.raises(MCPError) as exc_info:
        await library_agent.sync_github_resource("test_user", "test/repo")
    assert exc_info.value.code == -32603
    assert "GitHub API error" in exc_info.value.message
