from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["vial"])

logger = logging.getLogger(__name__)

@router.post("/manage")
async def manage_vial(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        op_type = operation.get("type")
        vial_id = operation.get("vial_id")
        if not op_type or not vial_id:
            raise ValueError("Operation type or vial_id missing")
        
        if op_type == "start":
            query = "UPDATE vials SET status = 'running' WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        elif op_type == "stop":
            query = "UPDATE vials SET status = 'stopped' WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        elif op_type == "configure":
            config = operation.get("config", {})
            query = "UPDATE vials SET config = config || $1::jsonb WHERE vial_id = $2 RETURNING config"
            result = await db.execute(query, config, vial_id)
        elif op_type == "reset":
            query = "UPDATE vials SET status = 'idle', quantum_state = NULL WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        else:
            raise ValueError("Unsupported vial operation")
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("vial_manager_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Vial operation validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("vial_manager_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in vial operation: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("vial_manager", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Vial operation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #manager #sqlite #octokit #neon_mcpfrom fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import sqlite3

router = APIRouter(prefix="/mcp/api/vial", tags=["vial"])

logger = logging.getLogger(__name__)

@router.post("/manage")
async def manage_vial(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        op_type = operation.get("type")
        vial_id = operation.get("vial_id")
        if not op_type or not vial_id:
            raise ValueError("Operation type or vial_id missing")
        
        if op_type == "start":
            query = "UPDATE vials SET status = 'running' WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        elif op_type == "stop":
            query = "UPDATE vials SET status = 'stopped' WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        elif op_type == "configure":
            config = operation.get("config", {})
            query = "UPDATE vials SET config = config || $1::jsonb WHERE vial_id = $2 RETURNING config"
            result = await db.execute(query, config, vial_id)
        elif op_type == "reset":
            query = "UPDATE vials SET status = 'idle', quantum_state = NULL WHERE vial_id = $1 RETURNING status"
            result = await db.execute(query, vial_id)
        else:
            raise ValueError("Unsupported vial operation")
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("vial_manager_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Vial operation validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("vial_manager_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in vial operation: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("vial_manager", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Vial operation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #vial #manager #sqlite #octokit #neon_mcp
