from fastapi import APIRouter, HTTPException, Depends
from ..security.octokit_oauth import get_octokit_auth
from ..error_logging.error_log import error_logger
from ..database import get_db
import logging
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["auth"])

logger = logging.getLogger(__name__)

@router.post("/auth/verify")
async def verify_vial_auth(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        wallet_address = operation.get("wallet_address")
        if not vial_id or not wallet_address:
            raise ValueError("Vial ID or wallet address missing")
        
        query = "SELECT wallet_address FROM vials WHERE vial_id = $1"
        result = await db.execute(query, vial_id)
        if not result or result[0]["wallet_address"] != wallet_address:
            raise ValueError("Wallet not linked to vial")
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "auth_verify", json.dumps({"wallet_address": wallet_address}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "verified": True}}
    except ValueError as e:
        error_logger.log_error("vial_auth_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Vial auth validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("vial_auth_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in vial auth: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("vial_auth", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Vial auth failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #auth #sqlite #octokit #neon_mcp
