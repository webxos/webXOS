# main/server/mcp/quantum/quantum_simulator.py
from qiskit import QuantumCircuit, Aer, execute
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from ..db.db_manager import DBManager
import os

app = FastAPI(title="Vial MCP Quantum Simulator")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class QuantumCircuitRequest(BaseModel):
    vial_id: str
    user_id: str
    circuit_data: Dict

class QuantumResultResponse(BaseModel):
    result_id: str
    vial_id: str
    results: Dict
    timestamp: str

class QuantumSimulator:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.backend = Aer.get_backend('statevector_simulator')

    def simulate_circuit(self, circuit_data: Dict) -> Dict:
        with self.metrics.track_span("simulate_circuit", {"circuit_data": circuit_data}):
            try:
                circuit = QuantumCircuit(circuit_data.get("num_qubits", 2))
                for gate in circuit_data.get("gates", []):
                    if gate == "H":
                        circuit.h(0)
                    elif gate == "CNOT":
                        circuit.cx(0, 1)
                    elif gate == "X":
                        circuit.x(0)
                    elif gate == "Z":
                        circuit.z(0)
                job = execute(circuit, self.backend)
                result = job.result()
                statevector = result.get_statevector(circuit)
                return {"statevector": statevector.to_dict()}
            except Exception as e:
                handle_generic_error(e, context="simulate_circuit")
                raise

quantum_simulator = QuantumSimulator()

@app.post("/quantum/simulate", response_model=QuantumResultResponse)
async def simulate_quantum_circuit(request: QuantumCircuitRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("simulate_quantum_circuit_endpoint", {"vial_id": request.vial_id}):
        try:
            metrics.verify_token(token)
            results = quantum_simulator.simulate_circuit(request.circuit_data)
            result_data = {
                "vial_id": request.vial_id,
                "user_id": request.user_id,
                "results": results,
                "timestamp": datetime.utcnow().isoformat()
            }
            result_id = db_manager.insert_one("quantum_results", result_data)
            return QuantumResultResponse(result_id=result_id, **result_data)
        except Exception as e:
            handle_generic_error(e, context="simulate_quantum_circuit_endpoint")
            raise HTTPException(status_code=500, detail=f"Failed to simulate circuit: {str(e)}")
