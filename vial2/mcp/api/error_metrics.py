from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["error_metrics"])

logger = logging.getLogger(__name__)

@router.get("/error/metrics")
async def get_error_metrics(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT event_type, COUNT(*) as count, MAX(created_at) as last_error FROM vial_logs GROUP BY event_type"
        metrics = await db.execute(query)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "error_metrics", json.dumps({"metrics_count": len(metrics)}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": metrics}}
    except sqlite3.Error as e:
        error_logger.log_error("error_metrics_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in error metrics: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("error_metrics", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Error metrics failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #error_metrics #sqlite #octokit #neon_mcp
