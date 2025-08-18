from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import sqlite3

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

        # Task handling for tools, resources, prompts
        if task_type == "tools/call":
            query = "SELECT * FROM vials WHERE vial_id=$1"
            result = await db.execute(query, params.get("vial_id"))
        elif task_type == "resources/get":
            query = "SELECT * FROM computes WHERE compute_id=$1"
            result = await db.execute(query, params.get("compute_id"))
        elif task_type == "prompts/list":
            query = "SELECT * FROM logs WHERE event_type='prompt'"
            result = await db.execute(query)
        else:
            raise ValueError("Unsupported task type")
        
        return result or {"status": "success", "data": []}
    except ValueError as e:
        error_logger.log_error("task_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=params)
        logger.error(f"Task validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params", "data": str(e)}
        })
    except sqlite3.Error as e:
        error_logger.log_error("task_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=params)
        logger.error(f"SQLite error in task handling: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": params}}
        })
    except Exception as e:
        error_logger.log_error("task_handling", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=params)
        logger.error(f"Task handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #tasks #sqlite #neon_mcp
