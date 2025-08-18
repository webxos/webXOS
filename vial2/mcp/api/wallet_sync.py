from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["wallet_sync"])

logger = logging.getLogger(__name__)

@router.post("/wallet/sync")
async def wallet_sync(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "SELECT wallet_address FROM vials WHERE vial_id = $1"
        result = await db.execute(query, vial_id)
        if not result or not result[0]["wallet_address"]:
            raise ValueError("Wallet address not found")
        
        wallet_data = {"address": result[0]["wallet_address"], "synced": True, "network": "webxos"}
        query = "UPDATE vials SET wallet_data = $1 WHERE vial_id = $2 RETURNING wallet_data"
        result = await db.execute(query, json.dumps(wallet_data), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "wallet_sync", json.dumps({"wallet_data": wallet_data}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("wallet_sync_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet sync validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("wallet_sync_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet sync: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("wallet_sync", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet sync failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #wallet #sqlite #octokit #neon_mcp
