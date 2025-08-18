from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import httpx
import logging
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["sync"])

logger = logging.getLogger(__name__)

@router.post("/sync")
async def sync_vial(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        node_id = operation.get("node_id")
        if not vial_id or not node_id:
            raise ValueError("Vial ID or node_id missing")
        
        # Fetch remote vial state
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.webxos.netlify.app/vial/status/{vial_id}", headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
            remote_state = response.json().get("result", {})
        
        # Sync local state
        query = "UPDATE vials SET status = $1, quantum_state = $2 WHERE vial_id = $3 RETURNING status, quantum_state"
        result = await db.execute(query, remote_state.get("status"), remote_state.get("quantum_state"), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "sync", str(remote_state), node_id))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("vial_sync_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Vial sync validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("vial_sync_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in vial sync: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("vial_sync", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Vial sync failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #sync #sqlite #octokit #neon_mcp
