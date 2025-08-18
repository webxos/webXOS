import subprocess
from fastapi import HTTPException
import logging
import os

logger = logging.getLogger(__name__)

async def execute_git_command(command: str, db):
    try {
        allowed_commands = ['status', 'pull', 'push', 'commit', 'branch', 'checkout', 'merge', 'clone']
        cmd_parts = command.strip().split()
        if not cmd_parts or cmd_parts[0] not in allowed_commands:
            raise ValueError(f"Unsupported git command: {command}")

        result = subprocess.run(
            ['git'] + cmd_parts,
            capture_output=True,
            text=True,
            cwd=os.getenv('REPO_PATH', './'),
            timeout=30
        )

        async with db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                "git_command",
                f"Git command executed: {command}\nOutput: {result.stdout}\nError: ${result.stderr}",
                datetime.utcnow()
            )

        return {
            "status": "success",
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        logger.error(f"Git command failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #git #neon_mcp
