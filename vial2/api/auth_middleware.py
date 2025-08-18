from fastapi import Request, HTTPException
from ..config.secrets import secrets_manager
from ..error_logging.error_log import error_logger
import logging
import jwt

logger = logging.getLogger(__name__)

class AuthMiddleware:
    async def __call__(self, request: Request, call_next):
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            token = auth_header.split(" ")[1]
            secret = secrets_manager.get_secret("JWT_SECRET_KEY")
            jwt.decode(token, secret, algorithms=["HS256"])
            return await call_next(request)
        except Exception as e:
            error_logger.log_error("auth_middleware", f"Authentication failed: {str(e)}", str(e.__traceback__))
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=401, detail=str(e))

auth_middleware = AuthMiddleware()

# xAI Artifact Tags: #vial2 #api #auth_middleware #neon_mcp
