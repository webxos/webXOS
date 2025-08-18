from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["mcp_endpoints"])

logger = logging.getLogger(__name__)

@router.post("/mcp/status")
async def mcp_status(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT vial_id, status, quantum_state, wallet_address FROM vials WHERE active = true"
        vials = await db.execute(query)
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_status", json.dumps({"vials_count": len(vials)}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": vials}}
    except sqlite3.Error as e:
        error_logger.log_error("mcp_status_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in MCP status: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("mcp_status", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"MCP status failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

@router.post("/mcp/connect")
async def mcp_connect(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        server = operation.get("server", "default")
        if not vial_id:
            raise ValueError("Vial ID missing")
        query = "UPDATE vials SET mcp_server = $1 WHERE vial_id = $2 RETURNING mcp_server"
        result = await db.execute(query, server, vial_id)
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "mcp_connect", json.dumps({"server": server}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("mcp_connect_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"MCP connect validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mcp_connect_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in MCP connect: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mcp_connect", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"MCP connect failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #endpoints #sqlite #octokit #neon_mcp
