from fastapi import APIRouter, HTTPException, Depends
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import uuid
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["api_key_generate"])

logger = logging.getLogger(__name__)

@router.post("/api_key/generate")
async def generate_api_key(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Missing vial_id")
        
        # Placeholder for Neon DB API key storage
        api_credentials = {
            "key": str(uuid.uuid4()),
            "secret": str(uuid.uuid4()),
            "created_at": int(time.time())
        }
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "api_key_generate", json.dumps({"key": api_credentials["key"]}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": api_credentials}}
    except ValueError as e:
        error_logger.log_error("api_key_generate_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"API key generation validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("api_key_generate_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in API key generation: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("api_key_generate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"API key generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #api_key #generate #sqlite #neon_mcp
