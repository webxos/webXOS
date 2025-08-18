from fastapi import APIRouter, Request, HTTPException
from ..error_logging.error_log import error_logger
import logging
import json
import sqlite3

router = APIRouter(prefix="/mcp/api", tags=["jsonrpc"])

logger = logging.getLogger(__name__)

@router.post("/jsonrpc")
async def handle_jsonrpc(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            raise HTTPException(status_code=400, detail={
                "jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None
            })
        
        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if not method:
            raise HTTPException(status_code=400, detail={
                "jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request_id
            })

        # Handle notifications (no id)
        is_notification = request_id is None

        # Map methods to internal logic
        from ..tasks import handle_task
        methods = {
            "tools/call": handle_task,
            "resources/get": handle_task,
            "prompts/list": handle_task
        }

        if method not in methods:
            raise HTTPException(status_code=400, detail={
                "jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": request_id
            })

        # Validate params
        if not isinstance(params, (dict, list)):
            raise HTTPException(status_code=400, detail={
                "jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params"}, "id": request_id
            })

        result = await methods[method](params)
        
        if is_notification:
            return None  # No response for notifications
        
        return {"jsonrpc": "2.0", "result": result, "id": request_id}
    except json.JSONDecodeError:
        error_logger.log_error("jsonrpc_parse", "Invalid JSON", "", sql_statement=None, sql_error_code=None, params=None)
        logger.error("Invalid JSON payload")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None
        })
    except sqlite3.Error as e:
        error_logger.log_error("jsonrpc_db", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=e.sqlite_errorcode, params=params)
        logger.error(f"SQLite error in JSON-RPC: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "params": params}}, "id": request_id
        })
    except Exception as e:
        error_logger.log_error("jsonrpc", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=params)
        logger.error(f"JSON-RPC handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}, "id": request_id
        })

# xAI Artifact Tags: #vial2 #api #jsonrpc #sqlite #neon_mcp
