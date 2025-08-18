from fastapi import APIRouter, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/api", tags=["health"])

@router.get("/health")
async def health_check(db=Depends(get_db)):
    try:
        async with db:
            await db.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        error_logger.log_error("health", f"Health check failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "database": "disconnected"}

# xAI Artifact Tags: #vial2 #api #health #neon_mcp
