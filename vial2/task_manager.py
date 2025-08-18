from fastapi import HTTPException
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def assign_task(vial_id: str, task_data: dict):
    try:
        db = await get_db()
        task_id = task_data.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID missing")
        await db.execute(
            "UPDATE vials SET tasks = tasks || $1::jsonb WHERE vial_id = $2",
            [task_data], vial_id
        )
        return {"status": "success", "task_id": task_id}
    except Exception as e:
        error_logger.log_error("task_manager", str(e), str(e.__traceback__))
        logger.error(f"Task assignment failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #task_manager #neon_mcp
