# main/server/mcp/agents/test_library_agent.py
import pytest
from pymongo import MongoClient
from ..agents.library_agent import LibraryAgent, MCPError
from ..agents.global_mcp_agents import GlobalMCPAgents

@pytest.fixture
def library_agent():
    agent = LibraryAgent()
    yield agent
    agent.collection.delete_many({})
    agent.close()

@pytest.fixture
def global_agents():
    agents = GlobalMCPAgents()
    yield agents
    agents.collection.delete_many({})
    agents.close()

@pytest.mark.asyncio
async def test_add_resource(library_agent, global_agents, mocker):
    # Mock resource API
    mocker.patch("requests.head", return_value=mocker.Mock(status_code=200))
    
    # Create agent
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]

    # Test adding resource
    result = await library_agent.add_resource(
        agent_id=agent_id,
        name="Test Dataset",
        uri="https://example.com/dataset",
        resource_type="dataset",
        metadata={"size": "1GB"},
        user_id="test_user"
    )
    assert result["status"] == "success"
    assert "resource_id" in result

@pytest.mark.asyncio
async def test_add_resource_invalid_agent(library_agent):
    with pytest.raises(MCPError) as exc_info:
        await library_agent.add_resource(
            agent_id="invalid_id",
            name="Test Dataset",
            uri="https://example.com/dataset",
            resource_type="dataset",
            metadata={"size": "1GB"},
            user_id="test_user"
        )
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Agent not found or access denied"

@pytest.mark.asyncio
async def test_add_resource_invalid_type(library_agent, global_agents):
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    with pytest.raises(MCPError) as exc_info:
        await library_agent.add_resource(
            agent_id=agent_id,
            name="Test Dataset",
            uri="https://example.com/dataset",
            resource_type="invalid",
            metadata={"size": "1GB"},
            user_id="test_user"
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Unsupported resource type"

@pytest.mark.asyncio
async def test_list_resources(library_agent, global_agents, mocker):
    mocker.patch("requests.head", return_value=mocker.Mock(status_code=200))
    
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    await library_agent.add_resource(
        agent_id=agent_id,
        name="Test Dataset",
        uri="https://example.com/dataset",
        resource_type="dataset",
        metadata={"size": "1GB"},
        user_id="test_user"
    )
    
    resources = await library_agent.list_resources(agent_id, "test_user", "dataset")
    assert len(resources) == 1
    assert resources[0]["name"] == "Test Dataset"
    assert resources[0]["uri"] == "https://example.com/dataset"
    assert resources[0]["type"] == "dataset"
    assert resources[0]["metadata"] == {"size": "1GB"}
