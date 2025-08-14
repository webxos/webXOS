# main/server/mcp/quantum/mcp_server_quantum.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from qiskit import QuantumCircuit, Aer, execute
from pymongo import MongoClient
import os
from datetime import datetime
from ..utils.error_handler import handle_generic_error
from ..utils.performance_metrics import PerformanceMetrics
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(title="Vial MCP Quantum Server")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["vial_mcp"]
quantum_collection = db["quantum_circuits"]
metrics = PerformanceMetrics()

class QuantumCircuitRequest(BaseModel):
    vial_id: str
    qubits: int
    circuit: str

class QuantumResult(BaseModel):
    vial_id: str
    result: dict
    execution_time: float
    timestamp: str

@app.post("/quantum/execute", response_model=QuantumResult)
async def execute_quantum_circuit(request: QuantumCircuitRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("execute_quantum_circuit") as span:
        try:
            metrics.verify_token(token)
            circuit = QuantumCircuit(request.qubits, request.qubits)
            # Simplified circuit parsing (in production, parse request.circuit safely)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure_all()
            
            backend = Aer.get_backend('qasm_simulator')
            start_time = datetime.utcnow().timestamp()
            job = execute(circuit, backend, shots=1024)
            result = job.result().get_counts()
            execution_time = datetime.utcnow().timestamp() - start_time
            
            quantum_collection.insert_one({
                "vial_id": request.vial_id,
                "circuit": request.circuit,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow()
            })
            
            span.set_attribute("vial_id", request.vial_id)
            span.set_attribute("qubits", request.qubits)
            return QuantumResult(
                vial_id=request.vial_id,
                result=result,
                execution_time=execution_time,
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            handle_generic_error(e, context="quantum_execution")
            raise HTTPException(status_code=500, detail=f"Quantum execution failed: {str(e)}")

@app.get("/quantum/history/{vial_id}")
async def get_quantum_history(vial_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_quantum_history") as span:
        try:
            metrics.verify_token(token)
            history = list(quantum_collection.find({"vial_id": vial_id}).sort("timestamp", -1).limit(10))
            span.set_attribute("vial_id", vial_id)
            return [{"vial_id": h["vial_id"], "result": h["result"], "execution_time": h["execution_time"], "timestamp": h["timestamp"]} for h in history]
        except Exception as e:
            handle_generic_error(e, context="quantum_history")
            raise HTTPException(status_code=500, detail=f"Failed to fetch quantum history: {str(e)}")
