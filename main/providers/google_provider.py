
from fastapi import APIRouter, HTTPException
from ..utils.logging import log_error, log_info

router = APIRouter()

@router.get("/google/health")
async def google_health():
    try:
        log_info("Google provider health check")
        return {"status": "healthy", "provider": "Google"}
    except Exception as e:
        log_error(f"Traceback: Google health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google error: {str(e)}")
