from fastapi import APIRouter, HTTPException, Depends
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["error_traceback"])

logger = logging.getLogger(__name__)

@router.get("/error/traceback")
async def get_error_traceback(token: str = Depends(get_octokit_auth)):
    try:
        with sqlite3.connect("error_log.db") as conn:
            cursor = conn.execute("SELECT error_id, error_type, error_message, traceback, created_at FROM error_logs ORDER BY created_at DESC LIMIT 10")
            errors = [{"error_id": row[0], "error_type": row[1], "error_message": row[2], "traceback": row[3], "created_at": row[4]} for row in cursor.fetchall()]
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": errors}}
    except sqlite3.Error as e:
        error_logger.log_error("error_traceback_db", str(e), str(e.__traceback__), sql_statement="SELECT FROM error_logs", sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in error traceback: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "SELECT FROM error_logs", "params": {}}}
        })
    except Exception as e:
        error_logger.log_error("error_traceback", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Error traceback retrieval failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #error #traceback #sqlite #neon_mcp
