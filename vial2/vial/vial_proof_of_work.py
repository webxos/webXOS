from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import hashlib
import time
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["proof_of_work"])

logger = logging.getLogger(__name__)

@router.post("/pow")
async def mine_proof_of_work(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        difficulty = operation.get("difficulty", 4)
        wallet_address = operation.get("wallet_address")
        if not vial_id or not wallet_address:
            raise ValueError("Vial ID or wallet address missing")
        
        # Validate wallet
        wallet_query = "SELECT balance FROM wallets WHERE address = $1"
        wallet = await db.execute(wallet_query, wallet_address)
        if not wallet:
            raise ValueError("Wallet not found")
        
        # Proof of work
        nonce = 0
        target = "0" * difficulty
        start_time = time.time()
        while True:
            hash_input = f"{vial_id}{wallet_address}{nonce}".encode()
            hash_result = hashlib.sha256(hash_input).hexdigest()
            if hash_result.startswith(target):
                break
            nonce += 1
            if time.time() - start_time > 60:  # Timeout after 60s
                raise ValueError("Proof of work timeout")
        
        # Update vial and wallet
        query = "UPDATE vials SET pow_result = $1 WHERE vial_id = $2 RETURNING pow_result"
        result = await db.execute(query, {"nonce": nonce, "hash": hash_result}, vial_id)
        wallet_update_query = "UPDATE wallets SET balance = balance + $1 WHERE address = $2"
        await db.execute(wallet_update_query, 1.0, wallet_address)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "pow", str(result), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "pow": result, "wallet_updated": True}}
    except ValueError as e:
        error_logger.log_error("pow_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Proof of work validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("pow_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in proof of work: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("pow", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Proof of work failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #proof_of_work #sqlite #octokit #neon_mcp
