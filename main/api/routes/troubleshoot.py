from fastapi import APIRouter, Depends
from ...security.authentication import verify_token

router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot(token: str = Depends(verify_token)):
    return {"message": "Diagnostics complete"}
