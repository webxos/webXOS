from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import subprocess

router = APIRouter(prefix="/mcp/api/vial", tags=["git"])

logger = logging.getLogger(__name__)

@router.post("/git")
async def git_operation(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        command = operation.get("command")
        if not vial_id or not command:
            raise ValueError("Vial ID or command missing")
        
        allowed_commands = ["git status", "git pull", "git push", "git commit -m"]
        if not any(command.startswith(cmd) for cmd in allowed_commands):
            raise ValueError("Unsupported Git command")
        
        try:
            # Sanitize command to prevent injection
            if "&&" in command or "|" in command:
                raise ValueError("Invalid characters in Git command")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except subprocess.SubprocessError as e:
            raise ValueError(f"Git command failed: {str(e)}")
        
        query = "UPDATE vials SET git_state = $1 WHERE vial_id = $2 RETURNING git_state"
        result_data = await db.execute(query, json.dumps(output), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "git_operation", json.dumps({"command": command, "output": output}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result_data}}
    except ValueError as e:
        error_logger.log_error("git_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Git operation validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("git_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in Git operation: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("git", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Git operation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #git #sqlite #octokit #neon_mcp
