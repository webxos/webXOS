import httpx
from ..config import Config
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def get_octokit_auth(token: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}", "Accept": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            scopes = response.headers.get("X-OAuth-Scopes", "").split(", ")
            return {"user": data["login"], "scopes": scopes}
    except Exception as e:
        error_logger.log_error("octokit_auth", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"token": token})
        logger.error(f"Octokit OAuth validation failed: {str(e)}")
        raise

async def get_web_flow_url():
    try:
        state = str(hash(str(time.time())))
        params = {
            "client_id": Config.GITHUB_CLIENT_ID,
            "scope": "repo user",
            "state": state,
            "redirect_uri": Config.GITHUB_REDIRECT_URI
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://github.com/login/oauth/authorize",
                params=params
            )
            return {"url": response.url, "state": state}
    except Exception as e:
        error_logger.log_error("octokit_web_flow", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=params)
        logger.error(f"Octokit web flow URL generation failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #security #octokit #oauth #neon_mcp
