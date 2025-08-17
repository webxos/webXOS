from fastapi import APIRouter, HTTPException, Depends
from ...config.mcp_config import config
from ...utils.logging import log_error, log_info
from .wallet import authenticate_token
import secrets

router = APIRouter()

@router.get("/generate-credentials")
async def generate_credentials(payload: dict = Depends(authenticate_token)):
    try:
        api_key = secrets.token_hex(16)
        api_secret = secrets.token_hex(32)
        credentials = {"key": api_key, "secret": api_secret}
        log_info(f"Credentials generated for client_id: {payload['sub']}")
        return credentials
    except Exception as e:
        log_error(f"Traceback: Credential generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credential error: {str(e)}")
