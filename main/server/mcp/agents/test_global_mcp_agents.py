import pytest
from fastapi.testclient import TestClient
from main.server.mcp.agents.global_mcp_agents import GlobalMCPAgents, AgentTaskRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def agent_manager():
    """Create a GlobalMCPAgents instance."""
    db_manager = DatabaseManager()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return GlobalMCPAgents(db_manager, security_manager, error_handler)

@pytest.mark.asyncio
async def test_execute_task_success(agent_manager, mocker):
    """Test successful task execution."""
    mocker.patch.object(agent_manager.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(agent_manager.agents["translator"], 'process_task', return_value={"translated": "Hello"})
    mocker.patch.object(agent_manager.db_manager, 'log_task', return_value="task_123")
    request = AgentTaskRequest(wallet_id="wallet_123", task_type="translator", parameters={"text": "Hola"})
    response = await agent_manager.execute_task(request, "mocked_token")
    assert response == {"task_id": "task_123", "result": {"translated": "Hello"}}

@pytest.mark.asyncio
async def test_execute_task_unauthorized(agent_manager, mocker):
    """Test task execution with unauthorized wallet."""
    mocker.patch.object(agent_manager.security_manager, 'validate_token', return_value={"wallet_id": "wrong_wallet"})
    request = AgentTaskRequest(wallet_id="wallet_123", task_type="translator", parameters={"text": "Hola"})
    with pytest.raises(HTTPException) as exc:
        await agent_manager.execute_task(request, "mocked_token")
    assert exc.value.status_code == 500
    assert "Unauthorized wallet access" in exc.value.detail

@pytest.mark.asyncio
async def test_execute_task_unknown_agent(agent_manager, mocker):
    """Test task execution with unknown agent type."""
    mocker.patch.object(agent_manager.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    request = AgentTaskRequest(wallet_id="wallet_123", task_type="unknown", parameters={"text": "Hola"})
    with pytest.raises(HTTPException) as exc:
        await agent_manager.execute_task(request, "mocked_token")
    assert exc.value.status_code == 500
    assert "Unknown agent type: unknown" in exc.value.detail
