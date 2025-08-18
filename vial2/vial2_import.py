from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["import"])

logger = logging.getLogger(__name__)

@router.post("/import")
async def import_wallet(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        import_data = operation.get("import_data")
        if not vial_id or not import_data:
            raise ValueError("Vial ID or import data missing")
        
        try:
            data = json.loads(import_data)
            wallet_address = data.get("wallet_address")
            balance = data.get("balance", 0.0)
            if not wallet_address or not isinstance(balance, (int, float)):
                raise ValueError("Invalid import data format")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON import data")
        
        query = "INSERT INTO wallets (address, balance, vial_id) VALUES ($1, $2, $3) ON CONFLICT (address) DO UPDATE SET balance = $2, vial_id = $3 RETURNING address, balance"
        result = await db.execute(query, wallet_address, balance, vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "wallet_import", json.dumps({"wallet_address": wallet_address, "balance": balance}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("import_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet import validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("import_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet import: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("import", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet import failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #import #sqlite #octokit #neon_mcp
