from fastapi import HTTPException
import logging
from .database import execute_query

logger = logging.getLogger(__name__)

async def get_status(db):
    try:
        vials = await execute_query(db, "SELECT * FROM vials")
        computes = await execute_query(db, "SELECT * FROM computes")
        logs = await execute_query(db, "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10")
        return {"vials": vials, "computes": computes, "logs": logs}
    except Exception as e:
        logger.error(f"Status fetch failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def replication_status(db):
    try:
        status = await execute_query(db, "SELECT * FROM pg_stat_replication")
        return {"replication_status": status}
    except Exception as e:
        logger.error(f"Replication status fetch failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #monitoring #neon_mcp
