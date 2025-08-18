from fastapi import APIRouter, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["health"])

logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check(db=Depends(get_db)):
    try:
        await db.execute("SELECT 1")
        return {"status": "healthy"}
    except Exception as e:
        error_logger.log_error("health_check", str(e), str(e.__traceback__))
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy"}

# xAI Artifact Tags: #vial2 #monitoring #health #neon_mcp
