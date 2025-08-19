import json
from fastapi import APIRouter
from ...mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/vial/agent/status")
async def get_agent_status():
    try:
        status = {
            "agent1": {"status": "running", "tasks": ["sync"]},
            "agent2": {"status": "stopped", "tasks": []}
        }
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": status}}
    except Exception as e:
        error_logger.log_error("agent_status", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Agent status fetch failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #netlify #agent #status #neon_mcp
