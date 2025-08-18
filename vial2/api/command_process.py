from fastapi import APIRouter, Depends, HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["commands"])

logger = logging.getLogger(__name__)

@router.post("/process")
async def process_api_command(command: dict, db=Depends(get_db)):
    try:
        cmd_type = command.get("type")
        allowed_commands = ["/prompt", "/task", "/config", "/status", "/git", "/configure", "/refresh_configuration", "/terminate_fast", "/help"]
        if cmd_type not in allowed_commands:
            raise HTTPException(status_code=400, detail="Invalid command type")
        await db.execute("INSERT INTO logs (event_type, message) VALUES ($1, $2)", "command", str(command))
        return {"status": "success", "command": cmd_type}
    except Exception as e:
        error_logger.log_error("command_process", str(e), str(e.__traceback__))
        logger.error(f"Command processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #command_process #neon_mcp
