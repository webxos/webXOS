# main/server/mcp/quantum/quantum_simulator.py
from typing import Dict, Any
from qiskit import QuantumCircuit, Aer, execute
from ..utils.mcp_error_handler import MCPError

class QuantumSimulator:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')

    async def simulate_circuit(self, circuit_data: Dict[str, Any], num_shots: int = 1024) -> Dict[str, Any]:
        try:
            if not circuit_data or "num_qubits" not in circuit_data or "gates" not in circuit_data:
                raise MCPError(code=-32602, message="Invalid circuit data: num_qubits and gates required")
            if num_shots < 1 or num_shots > 10000:
                raise MCPError(code=-32602, message="num_shots must be between 1 and 10000")

            circuit = QuantumCircuit(circuit_data["num_qubits"], circuit_data["num_qubits"])
            for gate in circuit_data["gates"]:
                if gate == "H":
                    circuit.h(range(circuit_data["num_qubits"]))
                elif gate == "CNOT":
                    for i in range(circuit_data["num_qubits"] - 1):
                        circuit.cx(i, i + 1)
                elif gate == "X":
                    circuit.x(range(circuit_data["num_qubits"]))
                else:
                    raise MCPError(code=-32602, message=f"Unsupported gate: {gate}")
            
            circuit.measure_all()
            job = execute(circuit, self.backend, shots=num_shots)
            result = job.result()
            counts = result.get_counts()

            return {
                "status": "success",
                "counts": counts,
                "circuit": circuit.qasm()
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Quantum simulation failed: {str(e)}")
