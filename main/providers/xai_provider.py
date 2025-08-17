
from fastapi import APIRouter, HTTPException
from ..utils.logging import log_error, log_info

router = APIRouter()

@router.get("/xai/health")
async def xai_health():
    try:
        log_info("xAI provider health check")
        return {"status": "healthy", "provider": "xAI"}
    except Exception as e:
        log_error(f"Traceback: xAI health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"xAI error: {str(e)}")
