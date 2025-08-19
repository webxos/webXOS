from fastapi import APIRouter, Depends
from mcp.security.auth_handler import get_current_user
from mcp.ui.terminal_interface import terminal_interface
from mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/control")
async def control_vial(command: str, user: dict = Depends(get_current_user)):
    try:
        output = terminal_interface.render(command)
        logger.info(f"Controlled vial with command {command} for user {user.get('id')}")
        return {"jsonrpc": "2.0", "result": {"output": output}}
    except Exception as e:
        error_logger.log_error("vial_control", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Vial control failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #api #controller #neon_mcp
