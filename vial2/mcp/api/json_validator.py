from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mcp.error_logging.error_log import error_logger
import logging
import json

router = APIRouter()

logger = logging.getLogger(__name__)

class JSONRequest(BaseModel):
    data: dict

@router.post("/validate")
async def validate_json(request: JSONRequest):
    try:
        json.loads(json.dumps(request.data))  # Validate JSON structure
        logger.info("JSON validated successfully")
        return {"jsonrpc": "2.0", "result": {"status": "valid"}}
    except Exception as e:
        error_logger.log_error("json_validate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"JSON validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={"jsonrpc": "2.0", "error": {"code": -32602, "message": str(e)}})

# xAI Artifact Tags: #vial2 #mcp #api #json #validator #neon_mcp
