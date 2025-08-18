from fastapi import APIRouter, HTTPException, Depends
from ...database import get_db
from ...error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import sqlite3
import json
import torch
import subprocess

router = APIRouter(prefix="/mcp/api/vial", tags=["training_pipeline"])

logger = logging.getLogger(__name__)

@router.post("/training/pipeline")
async def training_pipeline(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        vial_id = operation.get("vial_id")
        action = operation.get("action")
        if not vial_id or not action:
            raise ValueError("Vial ID or action missing")
        
        allowed_actions = ["start_training", "commit_training"]
        if action not in allowed_actions:
            raise ValueError("Unsupported training action")
        
        if action == "start_training":
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            input_data = torch.randn(100, 10)
            target = torch.randn(100, 1)
            loss = torch.nn.MSELoss()(model(input_data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state_dict = model.state_dict()
            query = "UPDATE vials SET quantum_state = $1 WHERE vial_id = $2 RETURNING quantum_state"
            result = await db.execute(query, {"loss": loss.item(), "model_state": state_dict}, vial_id)
        else:  # commit_training
            query = "SELECT quantum_state FROM vials WHERE vial_id = $1"
            result = await db.execute(query, vial_id)
            if not result or not result[0]["quantum_state"]:
                raise ValueError("No trained model found")
            commit_message = f"Training commit for vial {vial_id}"
            try:
                subprocess.run(f'git commit -m "{commit_message}"', shell=True, capture_output=True, text=True, timeout=30)
            except subprocess.SubprocessError as e:
                raise ValueError(f"Git commit failed: {str(e)}")
            result = {"commit_message": commit_message}
        
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO vial_logs (vial_id, event_type, event_data, node_id) VALUES (?, ?, ?, ?)",
                        (vial_id, f"training_{action}", json.dumps({"action": action, "result": result}), token.get("node_id", "unknown")))
        
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except ValueError as e:
        error_logger.log_error("training_pipeline_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Training pipeline validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except sqlite3.Error as e:
        error_logger.log_error("training_pipeline_db", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=operation)
        logger.error(f"SQLite error in training pipeline: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode, "sql_statement": query, "params": operation}}
        })
    except Exception as e:
        error_logger.log_error("training_pipeline", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=operation)
        logger.error(f"Training pipeline failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #training #sqlite #octokit #neon_mcp
