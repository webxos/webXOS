from fastapi import APIRouter, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["replication"])

logger = logging.getLogger(__name__)

@router.get("/replication_status")
async def get_replication_status(db=Depends(get_db)):
    try:
        status = await db.execute("SELECT * FROM pg_stat_replication")
        return {"replication_status": status}
    except Exception as e:
        error_logger.log_error("replication_status", str(e), str(e.__traceback__))
        logger.error(f"Replication status check failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #monitoring #replication_status #neon_mcp
