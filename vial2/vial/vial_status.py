from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["vial_status"])

logger = logging.getLogger(__name__)

@router.get("/status")
async def get_vial_status(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT vial_id, status, config FROM vials"
        result = await db.execute(query)
        with sqlite3.connect("error_log.db") as conn:
            log_count = conn.execute("SELECT COUNT(*) FROM vial_logs WHERE timestamp > datetime('now', '-1 hour')").fetchone()[0]
        return {"jsonrpc": "2.0", "result": {"status": "success", "vials": result, "log_count": log_count}}
    except sqlite3.Error as e:
        error_logger.log_error("vial_status_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=None)
        logger.error(f"SQLite error in vial status: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("vial_status", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Vial status retrieval failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #status #sqlite #octokit #neon_mcp
