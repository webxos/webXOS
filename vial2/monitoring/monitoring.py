from fastapi import APIRouter, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["monitoring"])

logger = logging.getLogger(__name__)

@router.get("/metrics")
async def get_system_metrics(db=Depends(get_db)):
    try:
        logs = await db.execute("SELECT event_type, COUNT(*) as count FROM logs GROUP BY event_type")
        computes = await db.execute("SELECT state, COUNT(*) as count FROM computes GROUP BY state")
        return {"logs": logs, "computes": computes}
    except Exception as e:
        error_logger.log_error("system_metrics", str(e), str(e.__traceback__))
        logger.error(f"Metrics retrieval failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #monitoring #metrics #neon_mcp
