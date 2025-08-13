import pytest
from vial.quantum_simulator import QuantumSimulator
from fastapi import HTTPException

@pytest.fixture
def simulator():
    return QuantumSimulator()

def test_simulate_success(simulator):
    result = simulator.simulate("user123", {"qubits": 2, "shots": 1024})
    assert result["status"] == "success"
    assert "counts" in result
    assert isinstance(result["counts"], dict)

def test_simulate_invalid_qubits(simulator):
    with pytest.raises(HTTPException) as exc:
        simulator.simulate("user123", {"qubits": 0, "shots": 1024})
    assert "Invalid qubits" in str(exc.value)

def test_simulate_invalid_shots(simulator):
    with pytest.raises(HTTPException) as exc:
        simulator.simulate("user123", {"qubits": 2, "shots": 0})
    assert "Invalid shots" in str(exc.value)

def test_simulate_logging(simulator, tmp_path):
    error_log = tmp_path / "errorlog.md"
    with open(error_log, "a") as f:
        f.write("")
    simulator.simulate("user123", {"qubits": 2, "shots": 1024})
    with open(error_log) as f:
        log_content = f.read()
    assert "Quantum simulation by user123" in log_content
