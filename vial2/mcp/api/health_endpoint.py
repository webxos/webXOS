from fastapi import APIRouter, Depends
from mcp.security.auth_handler import get_current_user
from mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check(user: dict = Depends(get_current_user)):
    try:
        logger.info("Health check passed")
        return {"jsonrpc": "2.0", "result": {"status": "healthy", "user": user.get("id")}}
    except Exception as e:
        error_logger.log_error("health_check", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Health check failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #api #health #endpoint #neon_mcp
