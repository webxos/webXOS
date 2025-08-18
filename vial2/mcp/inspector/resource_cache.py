from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import time

router = APIRouter(prefix="/mcp/api/vial", tags=["resource_cache"])

logger = logging.getLogger(__name__)

@router.post("/resource/cache")
async def cache_resource(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        resource_uri = operation.get("resource_uri")
        if not vial_id or not resource_uri:
            raise ValueError("Vial ID or resource URI missing")
        
        query = "SELECT resource_config FROM mcp_resources WHERE resource_uri = $1 AND active = true"
        resource = await db.execute(query, resource_uri)
        if not resource:
            raise ValueError("Resource not found")
        
        cache_data = {"resource_uri": resource_uri, "cached_at": int(time.time()), "resource_config": resource[0]["resource_config"]}
        query = "INSERT INTO resource_cache (vial_id, cache_data) VALUES ($1, $2) ON CONFLICT (vial_id, resource_uri) DO UPDATE SET cache_data = $2 RETURNING cache_data"
        result = await db.execute(query, vial_id, json.dumps(cache_data))
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "resource_cache", json.dumps({"resource_uri": resource_uri}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("resource_cache_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Resource cache validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("resource_cache_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in resource cache: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("resource_cache", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Resource cache failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #resource_cache #sqlite #octokit #neon_mcp
