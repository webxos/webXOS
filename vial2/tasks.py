from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def handle_task(params: dict):
    try:
        db = await get_db()
        task_type = params.get("type")
        if not task_type:
            raise ValueError("Task type missing")
        
        # Validate parameters
        if not isinstance(params, (dict, list)):
            raise ValueError("Invalid params format")

        # Example task handling for tools, resources, prompts
        if task_type == "tools/call":
            result = await db.execute("SELECT * FROM vials WHERE vial_id=$1", params.get("vial_id"))
        elif task_type == "resources/get":
            result = await db.execute("SELECT * FROM computes WHERE compute_id=$1", params.get("compute_id"))
        elif task_type == "prompts/list":
            result = await db.execute("SELECT * FROM logs WHERE event_type='prompt'")
        else:
            raise ValueError("Unsupported task type")
        
        return result or {"status": "success", "data": []}
    except ValueError as e:
        error_logger.log_error("task_validation", str(e), str(e.__traceback__))
        logger.error(f"Task validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params", "data": str(e)}
        })
    except Exception as e:
        error_logger.log_error("task_handling", str(e), str(e.__traceback__))
        logger.error(f"Task handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #tasks #neon_mcp
