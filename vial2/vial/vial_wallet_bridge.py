from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["wallet_bridge"])

logger = logging.getLogger(__name__)

@router.post("/wallet_bridge")
async def wallet_bridge(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        wallet_address = operation.get("wallet_address")
        action = operation.get("action")
        if not vial_id or not wallet_address or not action:
            raise ValueError("Vial ID, wallet address, or action missing")
        
        if action == "link":
            query = "UPDATE vials SET wallet_address = $1 WHERE vial_id = $2 RETURNING wallet_address"
            result = await db.execute(query, wallet_address, vial_id)
            with sqlite3.connect("error_log.db") as conn:
                conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                            (vial_id, "wallet_link", json.dumps({"wallet_address": wallet_address}), token.get("node_id", "unknown")))
        elif action == "unlink":
            query = "UPDATE vials SET wallet_address = NULL WHERE vial_id = $1 RETURNING wallet_address"
            result = await db.execute(query, vial_id)
            with sqlite3.connect("error_log.db") as conn:
                conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                            (vial_id, "wallet_unlink", json.dumps({"wallet_address": None}), token.get("node_id", "unknown")))
        elif action == "verify":
            query = "SELECT wallet_address FROM vials WHERE vial_id = $1"
            result = await db.execute(query, vial_id)
            if not result or result[0]["wallet_address"] != wallet_address:
                raise ValueError("Wallet not linked to vial")
            with sqlite3.connect("error_log.db") as conn:
                conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                            (vial_id, "wallet_verify", json.dumps({"wallet_address": wallet_address}), token.get("node_id", "unknown")))
        else:
            raise ValueError("Unsupported wallet bridge action")
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("wallet_bridge_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet bridge validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("wallet_bridge_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet bridge: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("wallet_bridge", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet bridge failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #wallet_bridge #sqlite #octokit #neon_mcp
