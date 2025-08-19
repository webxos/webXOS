from fastapi import Depends, HTTPException
from langchain.adapters.security import SecurityAdapter
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

def get_auth_token(token: str = Depends(lambda x: x.headers.get("Authorization"))):
    try:
        security_adapter = SecurityAdapter(api_key=os.getenv("AUTH_API_KEY", ""))
        if not token or not token.startswith("Bearer "):
            raise ValueError("Invalid or missing authentication token")
        if not security_adapter.validate_token(token.replace("Bearer ", "")):
            raise ValueError("Authentication failed")
        logger.info("Authentication successful with LangChain adapter")
        return token
    except ValueError as e:
        error_logger.log_error("auth_validation", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Authentication validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail={"jsonrpc": "2.0", "error": {"code": -32001, "message": str(e), "data": {}}})
    except Exception as e:
        error_logger.log_error("auth_handler", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}})

# xAI Artifact Tags: #vial2 #mcp #security #auth #handler #langchain #neon_mcp
