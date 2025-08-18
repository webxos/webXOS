from fastapi import APIRouter, Depends, HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["configuration"])

logger = logging.getLogger(__name__)

@router.post("/refresh_configuration")
async def refresh_configuration(db=Depends(get_db)):
    try:
        await db.execute("UPDATE computes SET last_activity = CURRENT_TIMESTAMP")
        return {"status": "success", "message": "Configuration refreshed"}
    except Exception as e:
        error_logger.log_error("refresh_configuration", str(e), str(e.__traceback__))
        logger.error(f"Configuration refresh failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #refresh_configuration #neon_mcp
