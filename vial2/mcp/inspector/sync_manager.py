from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["sync_manager"])

logger = logging.getLogger(__name__)

@router.post("/sync/state")
async def sync_state(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "SELECT quantum_state, wallet_data, git_model_state FROM vials WHERE vial_id = $1 AND active = true"
        state = await db.execute(query, vial_id)
        if not state:
            raise ValueError("Vial not found")
        
        sync_data = {
            "quantum_state": state[0]["quantum_state"],
            "wallet_data": state[0]["wallet_data"],
            "git_model_state": state[0]["git_model_state"],
            "synced_at": int(time.time())
        }
        query = "UPDATE vials SET last_sync = $1 WHERE vial_id = $2 RETURNING last_sync"
        result = await db.execute(query, json.dumps(sync_data), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "sync_state", json.dumps(sync_data), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("sync_state_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Sync state validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("sync_state_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in sync state: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("sync_state", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Sync state failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #sync #sqlite #octokit #neon_mcp
