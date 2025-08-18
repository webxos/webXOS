from fastapi import APIRouter, HTTPException
from ..error_logging.error_log import error_logger
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["offline"])

logger = logging.getLogger(__name__)

@router.post("/offline")
async def offline_fallback(operation: dict):
    try:
        vial_id = operation.get("vial_id")
        action = operation.get("action")
        if not vial_id or not action:
            raise ValueError("Vial ID or action missing")
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("PRAGMA busy_timeout = 5000")
            if action == "cache":
                conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                            (vial_id, "offline_cache", json.dumps(operation.get("data", {})), "offline_node"))
                conn.execute("CREATE TABLE IF NOT EXISTS offline_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, vial_id TEXT, action TEXT, data TEXT)")
                conn.execute("INSERT INTO offline_queue (vial_id, action, data) VALUES (?, ?, ?)",
                            (vial_id, action, json.dumps(operation.get("data", {}))))
            elif action == "sync":
                queue = conn.execute("SELECT id, action, data FROM offline_queue WHERE vial_id = ?", (vial_id,)).fetchall()
                for item in queue:
                    conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                                (vial_id, f"offline_sync_{item['action']}", item['data'], "offline_node"))
                    conn.execute("DELETE FROM offline_queue WHERE id = ?", (item["id"],))
            else:
                raise ValueError("Unsupported offline action")
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "action": action}}
    except ValueError as e:
        error_logger.log_error("offline_validation", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=None, params=operation)
        logger.error(f"Offline fallback validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("offline_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in offline fallback: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("offline", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Offline fallback failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #offline #sqlite #neon_mcp
