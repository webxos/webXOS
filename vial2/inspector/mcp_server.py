from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/inspector", tags=["mcp"])

logger = logging.getLogger(__name__)

@router.post("/tools/list")
async def list_tools(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT tool_name, tool_config FROM mcp_tools WHERE active = true"
        tools = await db.execute(query)
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_tools_list", json.dumps({"tools_count": len(tools)}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": tools}}
    except sqlite3.Error as e:
        error_logger.log_error("mcp_tools_list_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in tools list: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("mcp_tools_list", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Tools list failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

@router.post("/tools/call")
async def call_tool(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        tool_name = operation.get("tool_name")
        args = operation.get("args", {})
        if not tool_name:
            raise ValueError("Tool name missing")
        query = "SELECT tool_config FROM mcp_tools WHERE tool_name = $1 AND active = true"
        tool = await db.execute(query, tool_name)
        if not tool:
            raise ValueError("Tool not found")
        # Simulate tool execution (replace with actual tool logic)
        result = {"output": f"Executed {tool_name} with args {args}"}
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_tool_call", json.dumps({"tool_name": tool_name, "args": args}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("mcp_tool_call_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Tool call validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mcp_tool_call_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in tool call: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mcp_tool_call", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Tool call failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #inspector #sqlite #octokit #neon_mcp
