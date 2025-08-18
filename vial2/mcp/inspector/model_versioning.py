from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import torch

router = APIRouter(prefix="/mcp/api/vial", tags=["model_versioning"])

logger = logging.getLogger(__name__)

@router.post("/model/version")
async def model_version(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        action = operation.get("action")
        if not vial_id or not action:
            raise ValueError("Vial ID or action missing")
        
        allowed_actions = ["save_version", "load_version"]
        if action not in allowed_actions:
            raise ValueError("Unsupported model version action")
        
        if action == "save_version":
            query = "SELECT quantum_state FROM vials WHERE vial_id = $1"
            result = await db.execute(query, vial_id)
            if not result or not result[0]["quantum_state"]:
                raise ValueError("No model state found")
            model_state = result[0]["quantum_state"]["model_state"]
            version_data = {"model_state": model_state, "version": f"vial_{vial_id}_{int(time.time())}"}
            query = "INSERT INTO model_versions (vial_id, version_data) VALUES ($1, $2) RETURNING version_data"
            result = await db.execute(query, vial_id, json.dumps(version_data))
        else:  # load_version
            version_id = operation.get("version_id")
            if not version_id:
                raise ValueError("Version ID missing")
            query = "SELECT version_data FROM model_versions WHERE vial_id = $1 AND version_data->>'version' = $2"
            result = await db.execute(query, vial_id, version_id)
            if not result:
                raise ValueError("Version not found")
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, f"model_{action}", json.dumps({"action": action, "version_id": version_id if action == "load_version" else version_data["version"]}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("model_version_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Model version validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("model_version_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in model version: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("model_version", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Model version failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #model_versioning #sqlite #octokit #neon_mcp
