from fastapi import APIRouter, Depends
from utils.auth import verify_token

router = APIRouter()

@router.get("/quantum-link")
async def quantum_link(token: str = Depends(verify_token)):
    return {"status": "success", "quantum_state": {"qubits": [], "entanglement": "none"}}
