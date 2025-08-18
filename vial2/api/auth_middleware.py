from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from ..config import Config
from ..error_logging.error_log import error_logger
import jwt
import logging

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid or missing token")
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=["HS256"])
            request.state.user_id = payload.get("user_id")
            response = await call_next(request)
            return response
        except Exception as e:
            error_logger.log_error("auth_middleware", str(e), str(e.__traceback__))
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=401, detail=str(e))

# xAI Artifact Tags: #vial2 #api #auth_middleware #neon_mcp
