from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from ..error_logging.error_log import error_logger
import logging
import json

logger = logging.getLogger(__name__)

class JSONRequest(BaseModel):
    jsonrpc: str
    method: str
    params: dict
    id: str

class JSONHandler:
    def __init__(self):
        self.supported_methods = ["agent", "alert", "health"]

    def parse_json(self, raw_json: str):
        try:
            data = json.loads(raw_json)
            request = JSONRequest(**data)
            if request.jsonrpc != "2.0" or request.method not in self.supported_methods:
                raise ValueError("Invalid JSON-RPC request")
            logger.info(f"Parsed JSON request for method: {request.method}")
            return request
        except (json.JSONDecodeError, ValidationError) as e:
            error_logger.log_error("json_parse", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"JSON parsing failed: {str(e)}")
            raise HTTPException(status_code=400, detail={"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e), "data": {}}})
        except Exception as e:
            error_logger.log_error("json_handler", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"JSON handling failed: {str(e)}")
            raise HTTPException(status_code=500, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}})

json_handler = JSONHandler()

# xAI Artifact Tags: #vial2 #mcp #api #json #handler #neon_mcp
