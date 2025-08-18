from fastapi import APIRouter, HTTPException, Depends
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import re

router = APIRouter(prefix="/mcp/api/vial", tags=["console_commands"])

logger = logging.getLogger(__name__)

@router.post("/console/execute")
async def execute_console_command(command: dict, token: str = Depends(get_octokit_auth)):
    try:
        cmd_text = command.get("command")
        vial_id = command.get("vial_id", "vial1")
        if not cmd_text:
            raise ValueError("Command text missing")
        
        # Sanitize command
        cmd_text = re.sub(r'[<>{}\[\];]', '', cmd_text)
        if re.search(r'(system|admin|root|eval|exec)', cmd_text, re.IGNORECASE):
            raise ValueError("Command contains restricted keywords: system, admin, root, eval, exec")
        
        # Simulate command execution and resource connection
        resources = ["agent1", "llm_grok", "tool_quantum", "mcp_vial2"]  # Mock resources
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "console_command", json.dumps({"command": cmd_text, "resources": resources}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"output": f"Command executed: {cmd_text}", "connected_resources": resources}}}
    except ValueError as e:
        error_logger.log_error("console_command_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=command)
        logger.error(f"Console command validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": command}
        })
    except sqlite3.Error as e:
        error_logger.log_error("console_command_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=command)
        logger.error(f"SQLite error in console command: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": command}}
        })
    except Exception as e:
        error_logger.log_error("console_command", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=command)
        logger.error(f"Console command execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #console #commands #sqlite #neon_mcp
