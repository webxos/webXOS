from fastapi import APIRouter, HTTPException
from ..error_logging.error_log import error_logger
import logging
import sqlite3
import httpx

router = APIRouter(prefix="/mcp/api", tags=["log_aggregation"])

logger = logging.getLogger(__name__)

async def aggregate_logs_to_central(endpoint: str, db_path="error_log.db"):
    try:
        with sqlite3.connect(db_path) as conn:
            logs = conn.execute("SELECT * FROM errors WHERE timestamp > datetime('now', '-1 hour')").fetchall()
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json={"logs": logs})
                response.raise_for_status()
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("log_aggregator", str(e), str(e.__traceback__))
        logger.error(f"Log aggregation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/aggregate_logs")
async def aggregate_logs():
    try:
        central_endpoint = "https://central-logging.webxos.netlify.app/logs"
        result = await aggregate_logs_to_central(central_endpoint)
        return result
    except Exception as e:
        error_logger.log_error("aggregate_logs_endpoint", str(e), str(e.__traceback__))
        logger.error(f"Log aggregation endpoint failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #log_aggregator #sqlite #neon_mcp
