from fastapi import APIRouter, HTTPException, Depends
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import hashlib
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["mining_verification"])

logger = logging.getLogger(__name__)

@router.post("/mining/verify")
async def verify_mining(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        vial_id = operation.get("vial_id")
        nonce = operation.get("nonce")
        block_data = operation.get("block_data")
        if not vial_id or nonce is None or not block_data:
            raise ValueError("Missing vial_id, nonce, or block_data")
        
        # Sanitize block_data
        block_data = block_data[:1024]  # Limit to 1KB
        if '<script' in block_data.lower():
            raise ValueError("Block data contains invalid content")
        
        # Placeholder for Neon DB mining verification
        mining_data = {
            "hash": await sha256(f"{block_data}:{nonce}:{int(time.time())}"),
            "verified": True,
            "timestamp": int(time.time())
        }
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "mining_verify", json.dumps(mining_data), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": mining_data}}
    except ValueError as e:
        error_logger.log_error("mining_verify_validation", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=None, params=operation)
        logger.error(f"Mining verification validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mining_verify_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in mining verification: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mining_verify", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Mining verification failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

async def sha256(data):
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()

# xAI Artifact Tags: #vial2 #mcp #mining #verification #sqlite #neon_mcp
