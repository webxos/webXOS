import pytest
from fastapi.testclient import TestClient
from main.server.mcp.quantum.mcp_server_quantum import MCPQuantumHandler, QuantumRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.quantum_simulator import QuantumSimulator
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def quantum_handler():
    """Create an MCPQuantumHandler instance."""
    db_manager = DatabaseManager()
    quantum_simulator = QuantumSimulator()
    error_handler = ErrorHandler()
    return MCPQuantumHandler(db_manager, quantum_simulator, error_handler)

@pytest.mark.asyncio
async def test_process_quantum_success(quantum_handler, mocker):
    """Test successful quantum link processing."""
    mocker.patch.object(quantum_handler.quantum_simulator, 'simulate_quantum_link', return_value={"state": "entangled"})
    mocker.patch.object(quantum_handler.db_manager, 'add_quantum_link', return_value="quantum_123")
    request = QuantumRequest(wallet_id="wallet_123", vial_id="vial_456", db_type="postgres")
    response = await quantum_handler.process_quantum(request)
    assert response == {"quantum_id": "quantum_123", "vial_id": "vial_456", "result": {"state": "entangled"}}

@pytest.mark.asyncio
async def test_process_quantum_failure(quantum_handler, mocker):
    """Test quantum link processing failure."""
    mocker.patch.object(quantum_handler.quantum_simulator, 'simulate_quantum_link', side_effect=Exception("Simulation error"))
    request = QuantumRequest(wallet_id="wallet_123", vial_id="vial_456", db_type="postgres")
    with pytest.raises(HTTPException) as exc:
        await quantum_handler.process_quantum(request)
    assert exc.value.status_code == 500
    assert "Simulation error" in exc.value.detail
