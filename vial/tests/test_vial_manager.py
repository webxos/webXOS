import pytest
from vial.vial_manager import VialManager
from unittest.mock import patch
import os

@pytest.fixture
def manager():
    return VialManager()

def test_load_agents(manager):
    agents = manager.load_agents()
    assert set(agents.keys()) == {"nomic", "cognitallmware", "llmware", "jinaai"}
    assert all(hasattr(agent, "search") for agent in agents.values())

def test_load_tools(manager):
    tools = manager.load_tools()
    assert "sample_tool" in tools
    assert hasattr(tools["sample_tool"], "execute")

def test_execute_agent(manager):
    with patch("vial.agents.agent1.NomicAgent.search") as mock_search:
        mock_search.return_value = {"status": "success", "data": {"matches": []}}
        result = manager.execute_agent("nomic", "test query", "user123")
        assert result["status"] == "success"
        mock_search.assert_called_once_with("test query", "user123", 5)

def test_execute_tool(manager):
    with patch("vial.tools.sample_tool.SampleTool.execute") as mock_execute:
        mock_execute.return_value = {"status": "success", "data": {"output": "test"}}
        result = manager.execute_tool("sample_tool", {"input": "test"}, "user123")
        assert result["status"] == "success"
        mock_execute.assert_called_once_with("user123", {"input": "test"})

def test_execute_invalid_agent(manager):
    with pytest.raises(ValueError) as exc:
        manager.execute_agent("invalid_agent", "test query", "user123")
    assert "Unknown agent" in str(exc.value)

def test_execute_invalid_tool(manager):
    with pytest.raises(ValueError) as exc:
        manager.execute_tool("invalid_tool", {"input": "test"}, "user123")
    assert "Unknown tool" in str(exc.value)
