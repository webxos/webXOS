from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/inspector", tags=["mcp_resources"])

logger = logging.getLogger(__name__)

@router.post("/resources/list")
async def list_resources(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT resource_uri, resource_config FROM mcp_resources WHERE active = true"
        resources = await db.execute(query)
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_resources_list", json.dumps({"resources_count": len(resources)}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": resources}}
    except sqlite3.Error as e:
        error_logger.log_error("mcp_resources_list_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in resources list: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("mcp_resources_list", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Resources list failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

@router.post("/resources/get")
async def get_resource(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        resource_uri = operation.get("resource_uri")
        if not resource_uri:
            raise ValueError("Resource URI missing")
        query = "SELECT resource_config FROM mcp_resources WHERE resource_uri = $1 AND active = true"
        resource = await db.execute(query, resource_uri)
        if not resource:
            raise ValueError("Resource not found")
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_resource_get", json.dumps({"resource_uri": resource_uri}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": resource[0]}}
    except ValueError as e:
        error_logger.log_error("mcp_resource_get_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Resource get validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mcp_resource_get_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in resource get: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mcp_resource_get", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Resource get failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #inspector #resources #sqlite #octokit #neon_mcp
