from sqlalchemy.sql import text
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def execute_query(db, query: str, params: dict = None):
    try:
        async with db:
            result = await db.execute(text(query), params or {})
            if query.strip().upper().startswith("SELECT"):
                rows = await result.fetchall()
                return [dict(row) for row in rows]
            return {"status": "success"}
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def get_vial_status(db, vial_id: str):
    return await execute_query(db, "SELECT * FROM vials WHERE vial_id = :vial_id", {"vial_id": vial_id})

async def get_compute_status(db, compute_id: str):
    return await execute_query(db, "SELECT * FROM computes WHERE compute_id = :compute_id", {"compute_id": compute_id})

async def log_event(db, event_type: str, message: str):
    await execute_query(db, "INSERT INTO logs (event_type, message, timestamp) VALUES (:event_type, :message, CURRENT_TIMESTAMP)", {
        "event_type": event_type,
        "message": message
    })

# xAI Artifact Tags: #vial2 #sql #query_engine #neon_mcp
