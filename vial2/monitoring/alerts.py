from fastapi import APIRouter, HTTPException
import sqlite3
import httpx
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["alerts"])

logger = logging.getLogger(__name__)

@router.post("/alerts")
async def send_alerts():
    try:
        with sqlite3.connect("error_log.db") as conn:
            errors = conn.execute("SELECT * FROM errors WHERE timestamp > datetime('now', '-10 minutes') AND module LIKE '%error%'").fetchall()
            if errors:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://alerts.webxos.netlify.app/notify",
                        json={"alerts": [{"module": e[1], "message": e[2], "node_id": e[6]} for e in errors]}
                    )
                    response.raise_for_status()
        return {"status": "success", "alert_count": len(errors)}
    except Exception as e:
        error_logger.log_error("alerts", str(e), str(e.__traceback__), sql_statement="SELECT * FROM errors", sql_error_code=None, params=None)
        logger.error(f"Alert sending failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}
        })

# xAI Artifact Tags: #vial2 #monitoring #alerts #sqlite #neon_mcp
