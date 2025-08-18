from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["quantum_link"])

logger = logging.getLogger(__name__)

@router.post("/quantum/link")
async def quantum_link(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "UPDATE vials SET quantum_linked = true, quantum_state = $1 WHERE vial_id = $2 RETURNING quantum_state"
        quantum_state = {"linked": True, "network": "webxos_quantum"}
        result = await db.execute(query, json.dumps(quantum_state), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "quantum_link", json.dumps({"quantum_state": quantum_state}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("quantum_link_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Quantum link validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("quantum_link_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in quantum link: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("quantum_link", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Quantum link failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #quantum #sqlite #octokit #neon_mcp
