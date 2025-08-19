from fastapi import APIRouter, Depends
from mcp.security.auth_handler import get_current_user
from mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/help")
async def get_help(user: dict = Depends(get_current_user)):
    try:
        help_text = """
        Available Commands:
        - /control?command=status: Check system status
        - /multi_vial/train: Train multiple vials
        - /update: Update system
        """
        logger.info(f"Provided help for user {user.get('id')}")
        return {"jsonrpc": "2.0", "result": {"help": help_text}}
    except Exception as e:
        error_logger.log_error("help_endpoint", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Help endpoint failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #api #help #neon_mcp
