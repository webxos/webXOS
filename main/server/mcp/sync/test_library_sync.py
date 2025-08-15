# main/server/mcp/sync/test_library_sync.py
import pytest
from pymongo import MongoClient
from ..sync.library_sync import LibrarySync, MCPError
from ..agents.global_mcp_agents import GlobalMCPAgents
from datetime import datetime

@pytest.fixture
async def library_sync():
    sync = LibrarySync()
    yield sync
    sync.sync_log.delete_many({})
    sync.close()

@pytest.fixture
async def global_agents():
    agents = GlobalMCPAgents()
    yield agents
    agents.collection.delete_many({})
    agents.close()

@pytest.mark.asyncio
async def test_sync_resources(library_sync, global_agents, mocker):
    # Mock library agent
    mocker.patch.object(library_sync.library_agent, 'list_resources', return_value=[])
    mocker.patch.object(library_sync.library_agent, 'add_resource', return_value={"status": "success", "resource_id": "123"})
    
    # Create agent
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    # Test sync
    resources = [
        {
            "name": "Test Dataset",
            "uri": "https://example.com/dataset",
            "type": "dataset",
            "metadata": {"size": "1GB"}
        }
    ]
    result = await library_sync.sync_resources("test_user", agent_id, "external_service", resources)
    assert result["status"] == "success"
    assert result["added"] == 1
    assert result["updated"] == 0
    assert "sync_id" in result

@pytest.mark.asyncio
async def test_sync_resources_invalid(library_sync):
    with pytest.raises(MCPError) as exc_info:
        await library_sync.sync_resources("", "agent1", "external_service", [])
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "User ID, agent ID, and external service are required"

@pytest.mark.asyncio
async def test_get_sync_history(library_sync, global_agents, mocker):
    mocker.patch.object(library_sync.library_agent, 'list_resources', return_value=[])
    mocker.patch.object(library_sync.library_agent, 'add_resource', return_value={"status": "success", "resource_id": "123"})
    
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    await library_sync.sync_resources("test_user", agent_id, "external_service", [
        {"name": "Test Dataset", "uri": "https://example.com/dataset", "type": "dataset", "metadata": {"size": "1GB"}}
    ])
    
    history = await library_sync.get_sync_history("test_user", agent_id)
    assert len(history) == 1
    assert history[0]["external_service"] == "external_service"
    assert history[0]["added"] == 1
    assert history[0]["updated"] == 0
