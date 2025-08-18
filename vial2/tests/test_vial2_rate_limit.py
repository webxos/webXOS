from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["rate_limit"])

logger = logging.getLogger(__name__)

@router.post("/rate_limit")
async def manage_rate_limit(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        limit = operation.get("limit", 100)
        window = operation.get("window", 3600)  # Default 1 hour
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "INSERT OR REPLACE INTO rate_limits (vial_id, limit_count, window_seconds, last_reset) VALUES ($1, $2, $3, $4)"
        await db.execute(query, vial_id, limit, window, int(time.time()))
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "rate_limit_set", json.dumps({"limit": limit, "window": window}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"limit": limit, "window": window}}}
    except ValueError as e:
        error_logger.log_error("rate_limit_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Rate limit validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("rate_limit_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in rate limit: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("rate_limit", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Rate limit failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #rate_limit #sqlite #octokit #neon_mcp
