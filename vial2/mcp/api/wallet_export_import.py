from fastapi import APIRouter, HTTPException, Depends
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import uuid
import time
import re

router = APIRouter(prefix="/mcp/api/vial", tags=["wallet_export_import"])

logger = logging.getLogger(__name__)

@router.post("/wallet/export")
async def export_wallet(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        vial_id = operation.get("vial_id")
        if not vial_id:
            raise ValueError("Missing vial_id")
        
        # Placeholder for Neon DB wallet export
        data = {
            "network_id": str(uuid.uuid4()),
            "vials": [{"id": f"vial{i+1}", "wallet": {"address": str(uuid.uuid4()), "balance": 0.0, "hash": await sha256(f"vial{i+1}:{int(time.time())}")}} for i in range(4)],
            "connected_resources": ["agent1", "llm_grok", "tool_quantum", "mcp_vial2"]
        }
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "wallet_export", json.dumps(data), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": data}}
    except ValueError as e:
        error_logger.log_error("wallet_export_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet export validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("wallet_export_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet export: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("wallet_export", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet export failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

@router.post("/wallet/import")
async def import_wallet(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        vial_id = operation.get("vial_id")
        markdown = operation.get("markdown")
        if not vial_id or not markdown:
            raise ValueError("Missing vial_id or markdown")
        
        # Sanitize markdown
        markdown = markdown[:1024*1024]  # Limit to 1MB
        if '<script' in markdown.lower() or not re.match(r'# WebXOS Vial2 Wallet Export', markdown):
            raise ValueError("Invalid markdown content")
        
        # Placeholder for Neon DB wallet import
        data = {
            "wallet": {"address": str(uuid.uuid4()), "balance": 0.0, "hash": await sha256(f"{vial_id}:{markdown[:100]}:{int(time.time())}"), "reputation": 0},
            "vials": [{"id": f"vial{i+1}", "wallet": {"address": str(uuid.uuid4()), "balance": 0.0, "hash": await sha256(f"vial{i+1}:{int(time.time())}")}} for i in range(4)],
            "blockchain": [{"type": "import", "hash": await sha256(f"import:{vial_id}:{int(time.time())}")}],
            "connected_resources": ["agent1", "llm_grok", "tool_quantum", "mcp_vial2"]
        }
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "wallet_import", json.dumps(data), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": data}}
    except ValueError as e:
        error_logger.log_error("wallet_import_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet import validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("wallet_import_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in wallet import: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": "INSERT INTO vial_logs", "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("wallet_import", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Wallet import failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

async def sha256(data):
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()

# xAI Artifact Tags: #vial2 #mcp #wallet #export #import #sqlite #neon_mcp
