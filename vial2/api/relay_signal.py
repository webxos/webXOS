from fastapi import APIRouter, Depends, HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["relay"])

logger = logging.getLogger(__name__)

@router.post("/relay_signal")
async def relay_signal(signal: dict, db=Depends(get_db)):
    try:
        signal_type = signal.get("type")
        if not signal_type:
            raise HTTPException(status_code=400, detail="Signal type not provided")
        await db.execute(
            "INSERT INTO logs (event_type, message) VALUES ($1, $2)",
            "signal", str(signal)
        )
        return {"status": "success", "signal": signal_type}
    except Exception as e:
        error_logger.log_error("relay_signal", str(e), str(e.__traceback__))
        logger.error(f"Signal relay failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #relay_signal #neon_mcp
