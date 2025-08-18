from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json

router = APIRouter(prefix="/mcp/api/vial", tags=["prompt_template"])

logger = logging.getLogger(__name__)

@router.post("/prompt/template")
async def manage_prompt_template(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        prompt_name = operation.get("prompt_name")
        template = operation.get("template")
        if not vial_id or not prompt_name:
            raise ValueError("Vial ID or prompt name missing")
        
        query = "INSERT INTO prompt_templates (vial_id, prompt_name, template) VALUES ($1, $2, $3) ON CONFLICT (vial_id, prompt_name) DO UPDATE SET template = $3 RETURNING template"
        result = await db.execute(query, vial_id, prompt_name, json.dumps(template) if template else json.dumps({}))
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "prompt_template", json.dumps({"prompt_name": prompt_name}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("prompt_template_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Prompt template validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("prompt_template_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in prompt template: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("prompt_template", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Prompt template failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #prompt_template #sqlite #octokit #neon_mcp
