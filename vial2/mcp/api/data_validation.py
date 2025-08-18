from fastapi import HTTPException
from typing import Dict, Any
import re
import json
from ...mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def validate_data(data: Dict[str, Any]) -> bool:
    try:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        for key, value in data.items():
            if not re.match(r'^[a-zA-Z0-9_]+$', key):
                raise ValueError(f"Invalid key format: {key}")
            if isinstance(value, str) and '<script' in value.lower():
                raise ValueError("Data contains invalid content")
        return True
    except ValueError as e:
        error_logger.log_error("data_validation_value", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=data)
        logger.error(f"Data validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": data}
        })
    except Exception as e:
        error_logger.log_error("data_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=data)
        logger.error(f"Data validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #mcp #api #data #validation #neon_mcp
