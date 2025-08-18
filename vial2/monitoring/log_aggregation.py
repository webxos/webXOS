from fastapi import APIRouter, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["logs"])

logger = logging.getLogger(__name__)

@router.get("/logs")
async def get_aggregated_logs(db=Depends(get_db)):
    try:
        logs = await db.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100")
        return {"logs": logs}
    except Exception as e:
        error_logger.log_error("log_aggregation", str(e), str(e.__traceback__))
        logger.error(f"Log aggregation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #monitoring #log_aggregation #neon_mcp
