from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/inspector", tags=["mcp_prompts"])

logger = logging.getLogger(__name__)

@router.post("/prompts/list")
async def list_prompts(token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        query = "SELECT prompt_name, prompt_config FROM mcp_prompts WHERE active = true"
        prompts = await db.execute(query)
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_prompts_list", json.dumps({"prompts_count": len(prompts)}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": prompts}}
    except sqlite3.Error as e:
        error_logger.log_error("mcp_prompts_list_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params={})
        logger.error(f"SQLite error in prompts list: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query}}
        })
    except Exception as e:
        error_logger.log_error("mcp_prompts_list", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Prompts list failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

@router.post("/prompts/get")
async def get_prompt(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        prompt_name = operation.get("prompt_name")
        if not prompt_name:
            raise ValueError("Prompt name missing")
        query = "SELECT prompt_config FROM mcp_prompts WHERE prompt_name = $1 AND active = true"
        prompt = await db.execute(query, prompt_name)
        if not prompt:
            raise ValueError("Prompt not found")
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        ("inspector", "mcp_prompt_get", json.dumps({"prompt_name": prompt_name}), token.get("node_id", "unknown")))
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": prompt[0]}}
    except ValueError as e:
        error_logger.log_error("mcp_prompt_get_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Prompt get validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("mcp_prompt_get_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in prompt get: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("mcp_prompt_get", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Prompt get failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #inspector #prompts #sqlite #octokit #neon_mcp
