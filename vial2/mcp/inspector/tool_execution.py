from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["tool_execution"])

logger = logging.getLogger(__name__)

@router.post("/tool/execute")
async def execute_tool(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        tool_name = operation.get("tool_name")
        args = operation.get("args", {})
        if not vial_id or not tool_name:
            raise ValueError("Vial ID or tool name missing")
        
        query = "SELECT tool_config FROM mcp_tools WHERE tool_name = $1 AND active = true"
        tool = await db.execute(query, tool_name)
        if not tool:
            raise ValueError("Tool not found")
        
        # Simulate tool execution (replace with actual logic)
        execution_result = {"tool_name": tool_name, "args": args, "output": f"Executed {tool_name} for vial {vial_id}"}
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "tool_execute", json.dumps(execution_result), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": execution_result}}
    except ValueError as e:
        error_logger.log_error("tool_execute_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Tool execution validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("tool_execute_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in tool execution: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("tool_execute", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #tool_execution #sqlite #octokit #neon_mcp
