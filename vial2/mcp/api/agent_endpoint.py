from fastapi import APIRouter, HTTPException, Depends
from ..tools.agent_manager import agent_manager
from ..error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/mcp/api/vial/agent")
async def agent_request(agent_type: str, message: dict, credentials: dict = None):
    try:
        if not agent_type or not message.get("content"):
            raise ValueError("Agent type and message content required")
        await agent_manager.register_agent(agent_type, credentials or {})
        agent = await agent_manager.get_agent(agent_type)
        if not agent or not agent.get("credentials"):
            raise ValueError("Agent not registered or missing credentials")
        # Placeholder for agent logic (e.g., API call)
        response = f"Response from {agent_type}: {message['content']}"
        logger.info(f"Agent endpoint processed request for {agent_type}")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": response}}
    except ValueError as e:
        error_logger.log_error("agent_endpoint_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Agent endpoint validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={"jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": {}}})
    except Exception as e:
        error_logger.log_error("agent_endpoint", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Agent endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}})

# xAI Artifact Tags: #vial2 #mcp #api #agent #endpoint #neon_mcp
