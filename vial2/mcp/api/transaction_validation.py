from fastapi import HTTPException, Depends
from typing import Dict
from ...mcp.error_logging.error_log import error_logger
from ...security.octokit_oauth import get_octokit_auth
import logging
import re

logger = logging.getLogger(__name__)

def validate_transaction(transaction: Dict) -> bool:
    try:
        required_fields = ["id", "type", "timestamp", "amount"]
        if not all(field in transaction for field in required_fields):
            raise ValueError("Missing required transaction fields")
        if not re.match(r'^[a-zA-Z0-9-]+$', transaction["id"]):
            raise ValueError("Invalid transaction ID format")
        if transaction["amount"] < 0:
            raise ValueError("Amount cannot be negative")
        return True
    except ValueError as e:
        error_logger.log_error("transaction_validation_value", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=transaction)
        logger.error(f"Transaction validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": transaction}
        })
    except Exception as e:
        error_logger.log_error("transaction_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=transaction)
        logger.error(f"Transaction validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #api #transaction #validation #neon_mcp
