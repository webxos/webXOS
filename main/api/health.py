from fastapi import APIRouter
from ..utils.logging import log_error, log_info

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        log_info("Health check successful")
        return {"status": "ok", "balance": 0.0000, "reputation": 0}
    except Exception as e:
        log_error(f"Traceback: Health check failed: {str(e)}")
        return {"status": "error", "detail": str(e)}, 500
