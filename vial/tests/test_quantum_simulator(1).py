import pytest
from unittest.mock import patch
from vial.quantum_simulator import QuantumSimulator

@pytest.fixture
def quantum_simulator():
    return QuantumSimulator()

def test_initialize_network(quantum_simulator):
    network_id = "test-network"
    result = quantum_simulator.initialize_network(network_id)
    assert result["network_id"] == network_id
    assert len(result["vials"]) == 4
    for vial_id in result["vials"]:
        assert vial_id.startswith("vial")
        assert result["vials"][vial_id]["quantumState"] == {"qubits": [], "entanglement": "initialized"}

def test_simulate_quantum_state(quantum_simulator):
    vial_id = "vial1"
    quantum_state = quantum_simulator.simulate_quantum_state(vial_id)
    assert quantum_state["qubits"] == []
    assert quantum_state["entanglement"] in ["initialized", "trained", "offline"]

def test_simulate_quantum_state_invalid_vial(quantum_simulator):
    with pytest.raises(ValueError, match="Invalid vial ID"):
        quantum_simulator.simulate_quantum_state("invalid_vial")