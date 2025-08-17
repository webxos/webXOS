from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...mcp.handlers.resources import ResourceHandler

router = APIRouter()
resource_handler = ResourceHandler()

@router.get("/quantum-link")
async def quantum_link(user_id: str = Depends(verify_token)):
    try:
        state = await resource_handler.get_quantum_state()
        log_info(f"Quantum state retrieved for {user_id}")
        return {"status": "success", "quantum_state": state}
    except Exception as e:
        log_error(f"Quantum link failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum link error: {str(e)}")
