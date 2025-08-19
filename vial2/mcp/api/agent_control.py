from fastapi import APIRouter, HTTPException, Depends
from ...mcp.error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/vial/agent/control")
async def control_agent(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        agent_id = operation.get("agent_id")
        action = operation.get("action")
        if not agent_id or not action:
            raise ValueError("Missing agent_id or action")
        if action not in ["start", "stop", "restart"]:
            raise ValueError("Invalid action")
        result = {"agent_id": agent_id, "action": action, "status": "success"}
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("agent_control_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Agent control validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except Exception as e:
        error_logger.log_error("agent_control", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Agent control failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #api #agent #control #neon_mcp
