from fastapi import APIRouter, Depends, HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["mcp"])

logger = logging.getLogger(__name__)

@router.post("/endpoints")
async def process_command(command: dict, db=Depends(get_db)):
    try:
        cmd = command.get("command")
        if not cmd:
            raise HTTPException(status_code=400, detail="Command not provided")
        result = await db.execute("INSERT INTO logs (event_type, message) VALUES ($1, $2)", "command", cmd)
        return {"status": "success", "message": cmd}
    except Exception as e:
        error_logger.log_error("process_command", str(e), str(e.__traceback__))
        logger.error(f"Command processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
async def get_status(db=Depends(get_db)):
    try:
        status = await db.execute("SELECT * FROM computes")
        return {"computes": status}
    except Exception as e:
        error_logger.log_error("get_status", str(e), str(e.__traceback__))
        logger.error(f"Status retrieval failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #endpoints #neon_mcp
