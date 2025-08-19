from fastapi import APIRouter, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError
from ..tools.agent_manager import agent_manager
from ..security.audit import security_audit
from ..error_logging.error_log import error_logger
import logging
import json

router = APIRouter()

logger = logging.getLogger(__name__)

class AgentRequest(BaseModel):
    agent_type: str
    message: dict

@router.post("/mcp/api/vial/agent")
async def agent_request(request: AgentRequest):
    try:
        if not request.agent_type or not request.message.get("content"):
            raise ValueError("Agent type and message content required")
        if not await security_audit.check_input_sanitization(json.dumps(request.message)):
            raise ValueError("Input sanitization failed")
        await agent_manager.register_agent(request.agent_type)
        agent = await agent_manager.get_agent(request.agent_type)
        response = f"Response from {request.agent_type}: {request.message['content']}"
        encoded_response = jsonable_encoder({"jsonrpc": "2.0", "result": {"status": "success", "data": response}})
        logger.info(f"Agent endpoint processed JSON request for {request.agent_type}")
        return encoded_response
    except ValidationError as e:
        error_logger.log_error("agent_endpoint_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Agent endpoint validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={"jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": {}}})
    except Exception as e:
        error_logger.log_error("agent_endpoint", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Agent endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}})

# xAI Artifact Tags: #vial2 #mcp #api #agent #endpoint #json #neon_mcp
