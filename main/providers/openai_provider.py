from fastapi import APIRouter, HTTPException
from ..utils.logging import log_error, log_info

router = APIRouter()

@router.get("/openai/health")
async def openai_health():
    try:
        log_info("OpenAI provider health check")
        return {"status": "healthy", "provider": "OpenAI"}
    except Exception as e:
        log_error(f"Traceback: OpenAI health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
