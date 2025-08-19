from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class JSONResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: dict = None
    error: dict = None
    id: str

class JSONResponseHandler:
    def create_response(self, data: dict, request_id: str):
        try:
            response = JSONResponse(result=data if data else None, id=request_id)
            encoded_response = jsonable_encoder(response)
            logger.info(f"Created JSON response for id: {request_id}")
            return encoded_response
        except Exception as e:
            error_logger.log_error("json_response_create", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"JSON response creation failed: {str(e)}")
            return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}, "id": request_id}

json_response_handler = JSONResponseHandler()

# xAI Artifact Tags: #vial2 #mcp #api #json #response #neon_mcp
