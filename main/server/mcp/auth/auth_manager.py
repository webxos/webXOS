# main/server/mcp/auth/auth_manager.py
import jwt
import datetime
import logging
from typing import Dict, Optional
from ..utils.mcp_error_handler import MCPError
from ..utils.cache_manager import CacheManager
import os

logger = logging.getLogger("mcp")
cache = CacheManager()
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

class AuthManager:
    def __init__(self, user_data: Dict[str, Any]):
        self.user_data = user_data
        self.cache = cache

    async def authenticate(self, username: str, password: str) -> Dict[str, str]:
        try:
            if not username or not password:
                raise MCPError(code=-32602, message="Username and password are required")
            # Simulate authentication (replace with actual logic, e.g., database check)
            if username == "test_user" and password == "test_pass":
                token = jwt.encode({
                    "user_id": username,
                    "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
                }, SECRET_KEY, algorithm="HS256")
                await self.cache.set_cache(f"auth:{username}", {"token": token, "timestamp": datetime.datetime.utcnow().isoformat()})
                logger.info(f"User {username} authenticated at {datetime.datetime.utcnow().isoformat()}")
                return {"access_token": token, "redirect": "/dashboard"}
            raise MCPError(code=-32001, message="Invalid credentials")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Authentication failed: {str(e)}")

    async def verify_token(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            cached = await self.cache.get_cache(f"auth:{payload['user_id']}")
            return cached and cached["token"] == token
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise MCPError(code=-32002, message="Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise MCPError(code=-32002, message="Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Token verification failed: {str(e)}")

    async def logout(self, user_id: str) -> bool:
        try:
            await self.cache.delete_cache(f"auth:{user_id}")
            logger.info(f"User {user_id} logged out")
            return True
        except Exception as e:
            logger.error(f"Logout error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Logout failed: {str(e)}")
