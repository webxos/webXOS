from fastapi import APIRouter, HTTPException
from ..utils.logging import log_error, log_info

router = APIRouter()

@router.get("/anthropic/health")
async def anthropic_health():
    try:
        log_info("Anthropic provider health check")
        return {"status": "healthy", "provider": "Anthropic"}
    except Exception as e:
        log_error(f"Traceback: Anthropic health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anthropic error: {str(e)}")
