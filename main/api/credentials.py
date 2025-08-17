from fastapi import APIRouter, Depends, HTTPException
from ..security.security import verify_token
from ..utils.logging import log_error, log_info
import uuid

router = APIRouter()

@router.post("/generate-credentials")
async def generate_credentials(token: str = Depends(verify_token)):
    try:
        api_key = str(uuid.uuid4())
        api_secret = str(uuid.uuid4())
        log_info(f"Credentials generated: key={api_key[:8]}...")
        return {"key": api_key, "secret": api_secret}
    except Exception as e:
        log_error(f"Traceback: Credential generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credential error: {str(e)}")
