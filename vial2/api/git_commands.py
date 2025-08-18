from fastapi import APIRouter, HTTPException, Depends
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import subprocess

router = APIRouter(prefix="/mcp/api", tags=["git"])

logger = logging.getLogger(__name__)

@router.post("/git")
async def execute_git_command(command: dict, token: str = Depends(get_octokit_auth)):
    try:
        cmd_type = command.get("type")
        allowed_commands = ["status", "pull", "commit", "push"]
        if cmd_type not in allowed_commands:
            raise ValueError("Invalid Git command")
        
        # Execute Git command
        process = subprocess.run(["git", cmd_type], capture_output=True, text=True, check=True)
        result = {"status": "success", "output": process.stdout}
        
        # Log to SQLite
        error_logger.log_error("git_command", f"Git {cmd_type} executed", "", sql_statement=None, sql_error_code=None, params={"command": cmd_type, "user": token["user"]})
        return {"jsonrpc": "2.0", "result": result}
    except subprocess.CalledProcessError as e:
        error_logger.log_error("git_command_error", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"command": cmd_type})
        logger.error(f"Git command failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": e.stderr}
        })
    except Exception as e:
        error_logger.log_error("git_command", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"command": cmd_type})
        logger.error(f"Git command execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}
        })

# xAI Artifact Tags: #vial2 #api #git #octokit #sqlite #neon_mcp
