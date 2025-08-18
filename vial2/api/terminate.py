from fastapi import APIRouter, HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["terminate"])

logger = logging.getLogger(__name__)

@router.post("/terminate_fast")
async def terminate_fast(db=Depends(get_db)):
    try:
        await db.execute("UPDATE vials SET status = 'stopped' WHERE status = 'running'")
        return {"status": "success", "message": "Fast termination completed"}
    except Exception as e:
        error_logger.log_error("terminate_fast", str(e), str(e.__traceback__))
        logger.error(f"Fast termination failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/terminate_immediate")
async def terminate_immediate(db=Depends(get_db)):
    try:
        await db.execute("UPDATE vials SET status = 'stopped'")
        await db.execute("UPDATE computes SET state = 'Empty', readiness = FALSE")
        return {"status": "success", "message": "Immediate termination completed"}
    except Exception as e:
        error_logger.log_error("terminate_immediate", str(e), str(e.__traceback__))
        logger.error(f"Immediate termination failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #terminate #neon_mcp
