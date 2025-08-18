from fastapi import APIRouter, HTTPException, Depends
from ..config import Config
from ..error_logging.error_log import error_logger
import httpx
import logging

router = APIRouter(prefix="/mcp/api", tags=["auth"])

logger = logging.getLogger(__name__)

@router.post("/auth")
async def authenticate_oauth(code: str, redirect_uri: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.stack-auth.com/api/v1/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": Config.STACK_AUTH_CLIENT_ID,
                    "client_secret": Config.STACK_AUTH_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        error_logger.log_error("oauth_authentication", str(e), str(e.__traceback__))
        logger.error(f"OAuth authentication failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #security #oauth #neon_mcp
