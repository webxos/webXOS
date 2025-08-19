from fastapi import APIRouter, Depends
from mcp.langchain.agent_manager import agent_manager
from mcp.security.auth_handler import get_current_user
from mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/multi_vial/train")
async def train_multi_vial(data: dict, user: dict = Depends(get_current_user)):
    try:
        await agent_manager.train_agent("grok", data.get("training_data", []))
        logger.info(f"Trained multi-vial for user {user.get('id')}")
        return {"jsonrpc": "2.0", "result": {"status": "trained"}}
    except Exception as e:
        error_logger.log_error("multi_vial_train", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Multi-vial training failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #api #multi #vial #endpoint #neon_mcp
