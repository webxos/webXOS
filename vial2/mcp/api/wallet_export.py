from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["wallet_export"])

logger = logging.getLogger(__name__)

@router.post("/wallet/export")
async def wallet_export(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "SELECT wallet_data FROM vials WHERE vial_id = $1"
        result = await db.execute(query, vial_id)
        if not result or not result[0]["wallet_data"]:
            raise ValueError("Wallet data not found")
        
        wallet_data = result[0]["wallet_data"]
        export_data = {"vial_id": vial_id, "wallet_data": wallet_data, "exported_at": int(time.time())}
        query = "INSERT INTO wallet_exports (vial_id, export_data) VALUES ($1, $2) RETURNING export_data"
        result = await db.execute(query, vial_id, json.dumps(export_data))
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "wallet_export", json.dumps({"exported_at": export_data["exported_at"]}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("wallet_export_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet export validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("wallet_export_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet export: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("wallet_export", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet export failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #wallet #export #sqlite #octokit #neon_mcp
