import subprocess
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def execute_git_command(command: str):
    try:
        allowed_commands = ["status", "pull", "commit", "push"]
        cmd_parts = command.split()
        if cmd_parts[0] not in allowed_commands:
            raise HTTPException(status_code=400, detail="Unsupported Git command")
        process = subprocess.run(f"git {command}", shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            raise HTTPException(status_code=400, detail=process.stderr)
        return {"output": process.stdout}
    except Exception as e:
        error_logger.log_error("git_command", str(e), str(e.__traceback__))
        logger.error(f"Git command execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# xAI Artifact Tags: #vial2 #git #neon_mcp
