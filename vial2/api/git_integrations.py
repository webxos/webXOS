from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import subprocess

router = APIRouter(prefix="/mcp/api/vial", tags=["git_integration"])

logger = logging.getLogger(__name__)

@router.post("/git/model")
async def git_model_operation(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        action = operation.get("action")
        if not vial_id or not action:
            raise ValueError("Vial ID or action missing")
        
        allowed_actions = ["commit_model", "push_model", "pull_model"]
        if action not in allowed_actions:
            raise ValueError("Unsupported Git model action")
        
        try:
            if action == "commit_model":
                command = f'git commit -m "Model state update for vial {vial_id}"'
            elif action == "push_model":
                command = "git push"
            else:  # pull_model
                command = "git pull"
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except subprocess.SubprocessError as e:
            raise ValueError(f"Git model operation failed: {str(e)}")
        
        query = "UPDATE vials SET git_model_state = $1 WHERE vial_id = $2 RETURNING git_model_state"
        result_data = await db.execute(query, json.dumps(output), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "git_model_operation", json.dumps({"action": action, "output": output}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result_data}}
    except ValueError as e:
        error_logger.log_error("git_model_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Git model validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("git_model_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in Git model operation: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("git_model", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Git model operation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #git #sqlite #octokit #neon_mcp
