# main/server/mcp/agents/test_global_mcp_agents.py
import pytest
from pymongo import MongoClient
from ..agents.global_mcp_agents import GlobalMCPAgents, MCPError

@pytest.fixture
def agent_service():
    service = GlobalMCPAgents()
    yield service
    service.collection.delete_many({})
    service.close()

@pytest.mark.asyncio
async def test_create_agent(agent_service, mocker):
    mocker.patch.object(agent_service.wallet_service, 'create_wallet', return_value="0x1234567890abcdef")
    result = await agent_service.create_agent(
        vial_id="vial1",
        tasks=["train_model"],
        config={"lr": 0.01},
        user_id="test_user"
    )
    assert result["status"] == "success"
    assert "agent_id" in result
    assert result["wallet_address"] == "0x1234567890abcdef"

@pytest.mark.asyncio
async def test_create_agent_invalid_vial_id(agent_service):
    with pytest.raises(MCPError) as exc_info:
        await agent_service.create_agent(
            vial_id="invalid_id",
            tasks=["train_model"],
            config={"lr": 0.01},
            user_id="test_user"
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Invalid vial ID: Must start with 'vial'"

@pytest.mark.asyncio
async def test_update_agent_status(agent_service, mocker):
    mocker.patch.object(agent_service.wallet_service, 'create_wallet', return_value="0x1234567890abcdef")
    create_result = await agent_service.create_agent(
        vial_id="vial1",
        tasks=["train_model"],
        config={"lr": 0.01},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    result = await agent_service.update_agent_status(agent_id, "running", "test_user")
    assert result["status"] == "success"
    assert result["agent_id"] == agent_id

@pytest.mark.asyncio
async def test_update_agent_status_invalid(agent_service):
    with pytest.raises(MCPError) as exc_info:
        await agent_service.update_agent_status("invalid_id", "running", "test_user")
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Agent not found or access denied"

@pytest.mark.asyncio
async def test_list_agents(agent_service, mocker):
    mocker.patch.object(agent_service.wallet_service, 'create_wallet', return_value="0x1234567890abcdef")
    await agent_service.create_agent(
        vial_id="vial1",
        tasks=["train_model"],
        config={"lr": 0.01},
        user_id="test_user"
    )
    agents = await agent_service.list_agents("test_user")
    assert len(agents) == 1
    assert agents[0]["vial_id"] == "vial1"
    assert agents[0]["user_id"] == "test_user"
    assert agents[0]["wallet_address"] == "0x1234567890abcdef"
