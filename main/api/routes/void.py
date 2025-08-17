from fastapi import APIRouter, HTTPException, Depends
from ...utils.logging import log_error, log_info
from .wallet import authenticate_token

router = APIRouter()

@router.post("/void")
async def void_transaction(payload: dict = Depends(authenticate_token)):
    try:
        log_info(f"Transaction voided for client_id: {payload['sub']}")
        return {"status": "success", "message": "Transaction voided"}
    except Exception as e:
        log_error(f"Traceback: Void transaction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Void error: {str(e)}")
