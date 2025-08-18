from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/inspector", tags=["mcp_tools"])

logger = logging.getLogger(__name__)

@router.post("/tools/register")
async def register_tool(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        tool_name = operation.get("tool_name")
        tool_config = operation.get("tool_config")
        if not tool_name or not tool_config:
            raise ValueError("Tool name or config missing")
        query = "INSERT INTO mcp_tools (tool_name, tool_config, active) VALUES ($1, $2, true) ON CONFLICT (tool_name) DO UPDATE SET tool_config = $2, active = true"
        await db.execute(query, tool_name, json.dumps(tool_config))
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_tool_register", json.dumps({"tool_name": tool_name}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"tool_name": tool_name}}}
    except ValueError as e:
        error_logger.log_error("mcp_tool_register_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Tool register validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mcp_tool_register_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in tool register: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mcp_tool_register", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Tool register failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #inspector #tools #sqlite #octokit #neon_mcp
