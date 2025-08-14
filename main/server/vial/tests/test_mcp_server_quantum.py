import pytest,asyncio
from fastapi.testclient import TestClient
from main.server.unified_server import app
from main.server.mcp.mcp_server_quantum import MCPQuantumHandler
from main.server.mcp.mcp_auth_server import MCPAuthServer

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def quantum_handler():
    """Create an MCPQuantumHandler instance."""
    return MCPQuantumHandler()

@pytest.mark.asyncio
async def test_quantum_process_success(client,quantum_handler,mocker):
    """Test successful quantum processing with valid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=True)
    mocker.patch("main.server.quantum_simulator.QuantumSimulator.process_quantum_link",return_value={
        "vial_id":"vial_1",
        "state":[[0.1+0.2j],[0.3+0.4j]],
        "timestamp":"2025-08-13T21:27:00Z"
    })
    mocker.patch("sqlite3.connect",autospec=True)
    response=client.post("/api/quantum/link",
                         json={"vial_id":"vial_1","prompt":"Test prompt","wallet_id":"wallet_123"},
                         headers={"Authorization":"Bearer test_token"})
    assert response.status_code==200
    assert response.json()["status"]=="success"
    assert response.json()["quantum_state"]["vial_id"]=="vial_1"
    assert response.json()["wallet_id"]=="wallet_123"

@pytest.mark.asyncio
async def test_quantum_process_invalid_token(client,quantum_handler,mocker):
    """Test quantum processing with invalid token."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=False)
    response=client.post("/api/quantum/link",
                         json={"vial_id":"vial_1","prompt":"Test prompt","wallet_id":"wallet_123"},
                         headers={"Authorization":"Bearer invalid_token"})
    assert response.status_code==401
    assert response.json()["detail"]=="Invalid access token"

@pytest.mark.asyncio
async def test_quantum_process_failure(client,quantum_handler,mocker):
    """Test quantum processing failure due to internal error."""
    mocker.patch.object(MCPAuthServer,'verify_oauth_token',return_value=True)
    mocker.patch("main.server.quantum_simulator.QuantumSimulator.process_quantum_link",side_effect=Exception("Quantum error"))
    response=client.post("/api/quantum/link",
                         json={"vial_id":"vial_1","prompt":"Test prompt","wallet_id":"wallet_123"},
                         headers={"Authorization":"Bearer test_token"})
    assert response.status_code==500
    assert "Quantum error" in response.json()["detail"]