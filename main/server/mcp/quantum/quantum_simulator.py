# main/server/mcp/quantum/quantum_simulator.py
from qiskit import QuantumCircuit, Aer, execute
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from ..db.db_manager import DBManager
from fastapi.security import OAuth2PasswordBearer
import os

app = FastAPI(title="Vial MCP Quantum Simulator")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class QuantumCircuitRequest(BaseModel):
    vial_id: str
    circuit_data: Dict
    user_id: str

class QuantumCircuitResponse(BaseModel):
    vial_id: str
    result: Dict
    execution_time: float

@app.post("/quantum/simulate", response_model=QuantumCircuitResponse)
async def simulate_circuit(request: QuantumCircuitRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("simulate_circuit", {"vial_id": request.vial_id, "user_id": request.user_id}):
        try:
            metrics.verify_token(token)
            circuit = QuantumCircuit.from_qasm_str(request.circuit_data.get("qasm", ""))
            backend = Aer.get_backend("qasm_simulator")
            start_time = datetime.utcnow()
            job = execute(circuit, backend, shots=1024)
            result = job.result().to_dict()
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            db_manager.insert_one("quantum_circuits", {
                "vial_id": request.vial_id,
                "circuit": request.circuit_data.get("qasm", ""),
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow(),
                "user_id": request.user_id
            })
            
            return QuantumCircuitResponse(
                vial_id=request.vial_id,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            handle_generic_error(e, context="simulate_circuit")
            raise HTTPException(status_code=500, detail=f"Failed to simulate circuit: {str(e)}")
