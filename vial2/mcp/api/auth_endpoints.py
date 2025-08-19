from fastapi import APIRouter, Depends
from mcp.security.auth_handler import get_current_user
from mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/auth")
async def authenticate(user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Authenticated user {user.get('id')}")
        return {"jsonrpc": "2.0", "result": {"status": "authenticated", "user": user.get("id")}}
    except Exception as e:
        error_logger.log_error("auth_endpoint", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Authentication endpoint failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #api #auth #endpoint #neon_mcp
