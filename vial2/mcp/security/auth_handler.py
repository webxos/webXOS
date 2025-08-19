from fastapi import HTTPException, Depends
from mcp.security.audit_logger import log_audit_event
from mcp.error_logging.error_log import error_logger
import logging
import netlify_oauth2

logger = logging.getLogger(__name__)

async def get_current_user(token: str = Depends(netlify_oauth2.get_token)):
    try:
        oauth = netlify_oauth2.OAuth2(token=token)
        user = oauth.get_user_info()
        if not user.get("id"):
            raise ValueError("Invalid user ID")
        await log_audit_event("auth_success", {"user": user.get("id"), "token": token[:8] + "..."})  # Partial token log
        logger.info(f"Authenticated user {user.get('id')}")
        return user
    except Exception as e:
        error_logger.log_error("auth_handler", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

# xAI Artifact Tags: #vial2 #mcp #security #auth #handler #neon_mcp
