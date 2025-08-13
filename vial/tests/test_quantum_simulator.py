import pytest
from vial.quantum_simulator import QuantumSimulator

@pytest.fixture
def quantum_simulator():
    return QuantumSimulator()

def test_simulate_task(quantum_simulator):
    result = quantum_simulator.simulate_task("vial1", {"task": "test"})
    assert result["success"]
    assert "task_id" in result["state"]
    assert result["state"]["entanglement"] == "synced"

def test_validate_quantum_state(quantum_simulator):
    valid_state = {"qubits": [], "entanglement": "synced"}
    assert quantum_simulator.validate_quantum_state(valid_state)
    invalid_state = {"qubits": [], "entanglement": "unsynced"}
    assert not quantum_simulator.validate_quantum_state(invalid_state)
    invalid_state = {"qubits": None}
    assert not quantum_simulator.validate_quantum_state(invalid_state)
