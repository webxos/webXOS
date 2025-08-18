from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time
import hashlib

router = APIRouter(prefix="/mcp/api/vial", tags=["mining"])

logger = logging.getLogger(__name__)

@router.post("/mining/start")
async def start_mining(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "SELECT wallet_data FROM vials WHERE vial_id = $1"
        result = await db.execute(query, vial_id)
        if not result or not result[0]["wallet_data"]:
            raise ValueError("Wallet data not found")
        
        nonce = 0
        difficulty = 4
        block_data = f"{vial_id}:{time.time()}"
        while True:
            hash_input = f"{block_data}:{nonce}".encode()
            hash_result = hashlib.sha256(hash_input).hexdigest()
            if hash_result.startswith("0" * difficulty):
                break
            nonce += 1
        
        mining_result = {"vial_id": vial_id, "nonce": nonce, "hash": hash_result, "timestamp": int(time.time())}
        query = "INSERT INTO mining_records (vial_id, mining_data) VALUES ($1, $2) RETURNING mining_data"
        result = await db.execute(query, vial_id, json.dumps(mining_result))
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "mining_start", json.dumps(mining_result), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("mining_start_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Mining start validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mining_start_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in mining start: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mining_start", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Mining start failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #mining #sqlite #octokit #neon_mcp
