# main/server/mcp/agents/test_global_mcp_agents.py
import pytest
from ..agents.global_mcp_agents import GlobalMCPAgents, MCPError
from ..utils.performance_metrics import PerformanceMetrics

@pytest.fixture
async def global_agents():
    agents = GlobalMCPAgents()
    yield agents
    agents.agents.delete_many({})
    agents.db["sub_issues"].delete_many({})
    agents.close()

@pytest.mark.asyncio
async def test_create_agent(global_agents):
    result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    assert "agent_id" in result
    assert result["status"] == "created"
    agent = global_agents.agents.find_one({"agent_id": result["agent_id"]})
    assert agent["user_id"] == "test_user"
    assert agent["vial_id"] == "vial1"

@pytest.mark.asyncio
async def test_create_agent_invalid(global_agents):
    with pytest.raises(MCPError) as exc_info:
        await global_agents.create_agent("", [], {}, "test_user")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Vial ID, tasks, and user ID are required"

@pytest.mark.asyncio
async def test_execute_workflow(global_agents, mocker):
    mocker.patch.object(global_agents.library_agent, "list_resources", return_value=[{"uri": "test_resource"}])
    mocker.patch.object(global_agents.translator_agent, "translate_config", return_value={"translated": True})
    
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["manage_resources", "translate"],
        config={"resource_type": "dataset"},
        user_id="test_user"
    )
    result = await global_agents.execute_workflow(
        agent_id=create_result["agent_id"],
        workflow_config={"action": "process_data"},
        user_id="test_user"
    )
    assert "workflow_id" in result
    assert result["status"] == "executed"
    assert "resources" in result["config"]

@pytest.mark.asyncio
async def test_add_sub_issue(global_agents):
    result = await global_agents.add_sub_issue("parent123", "Sub-issue content", "test_user")
    assert "sub_issue_id" in result
    assert result["status"] == "created"
    sub_issue = global_agents.db["sub_issues"].find_one({"sub_issue_id": result["sub_issue_id"]})
    assert sub_issue["parent_issue_id"] == "parent123"
    assert sub_issue["user_id"] == "test_user"
