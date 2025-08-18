from fastapi import APIRouter, HTTPException, Depends
from ..config import Config
from ..error_logging.error_log import error_logger
from .octokit_oauth import get_octokit_auth
import logging
import httpx

router = APIRouter(prefix="/mcp/api", tags=["auth"])

logger = logging.getLogger(__name__)

@router.post("/auth")
async def authenticate_oauth(code: str, redirect_uri: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": Config.GITHUB_CLIENT_ID,
                    "client_secret": Config.GITHUB_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri
                },
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise HTTPException(status_code=400, detail=data["error"])
            # Use Octokit-inspired token validation
            octokit_auth = await get_octokit_auth(data["access_token"])
            return {"token": data["access_token"], "scopes": octokit_auth.get("scopes", [])}
    except Exception as e:
        error_logger.log_error("oauth_authentication", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"code": code})
        logger.error(f"OAuth authentication failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"params": {"code": code}}}
        })

# xAI Artifact Tags: #vial2 #security #oauth #octokit #neon_mcp
