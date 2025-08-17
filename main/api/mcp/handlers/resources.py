from fastapi import Depends
from utils.auth import verify_token

async def get_quantum_state(token: str = Depends(verify_token)):
    return {"qubits": [], "entanglement": "none"}
