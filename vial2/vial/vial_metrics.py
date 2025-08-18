from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["metrics"])

logger = logging.getLogger(__name__)

@router.get("/metrics")
async def get_vial_metrics(vial_id: str = None, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        start_time = time.time()
        if vial_id:
            query = "SELECT status, quantum_state, pow_result FROM vials WHERE vial_id = $1"
            vial_data = await db.execute(query, vial_id)
        else:
            query = "SELECT vial_id, status, quantum_state, pow_result FROM vials"
            vial_data = await db.execute(query)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("PRAGMA busy_timeout = 5000")
            log_query = "SELECT event_type, COUNT(*) as count FROM vial_logs WHERE timestamp > datetime('now', '-1 hour') GROUP BY event_type"
            log_metrics = conn.execute(log_query).fetchall()
        
        latency = time.time() - start_time
        error_logger.log_error("metrics_success", "Metrics retrieved", "", sql_statement=query, sql_error_code=None, params={"vial_id": vial_id, "latency": latency})
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "vials": vial_data, "log_metrics": log_metrics, "latency": latency}}
    except sqlite3.Error as e:
        error_logger.log_error("metrics_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={"vial_id": vial_id})
        logger.error(f"SQLite error in metrics: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": {"vial_id": vial_id}}}
        })
    except Exception as e:
        error_logger.log_error("metrics", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"vial_id": vial_id})
        logger.error(f"Metrics retrieval failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #metrics #sqlite #octokit #neon_mcp
