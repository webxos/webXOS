from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import torch

router = APIRouter(prefix="/mcp/api/vial", tags=["pytorch_optimizer"])

logger = logging.getLogger(__name__)

@router.post("/optimizer/config")
async def configure_optimizer(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        optimizer_type = operation.get("optimizer_type", "Adam")
        learning_rate = operation.get("learning_rate", 0.001)
        if not vial_id:
            raise ValueError("Vial ID missing")
        
        query = "SELECT quantum_state FROM vials WHERE vial_id = $1"
        result = await db.execute(query, vial_id)
        if not result or not result[0]["quantum_state"]:
            raise ValueError("No model state found")
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        model.load_state_dict(result[0]["quantum_state"]["model_state"])
        
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type")
        
        optimizer_config = {"type": optimizer_type, "learning_rate": learning_rate}
        query = "UPDATE vials SET optimizer_config = $1 WHERE vial_id = $2 RETURNING optimizer_config"
        result = await db.execute(query, json.dumps(optimizer_config), vial_id)
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, "optimizer_config", json.dumps(optimizer_config), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("optimizer_config_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Optimizer config validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("optimizer_config_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in optimizer config: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("optimizer_config", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Optimizer config failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #pytorch #optimizer #sqlite #octokit #neon_mcp
