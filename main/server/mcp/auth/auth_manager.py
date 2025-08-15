# main/server/mcp/auth/auth_manager.py
import jwt
import datetime
import logging
import os
from typing import Dict, Optional, List
from ..utils.mcp_error_handler import MCPError
from ..utils.cache_manager import CacheManager
from pathlib import Path

logger = logging.getLogger("mcp")
cache = CacheManager()
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

class AuthManager:
    def __init__(self, user_data: Dict[str, Any]):
        self.user_data = user_data
        self.cache = cache
        self.required_files = [
            "main/server/mcp/unified_server.py",
            "main/server/mcp/auth/auth_manager.py",
            "main/server/mcp/api_gateway/service_registry.py",
            "main/server/mcp/utils/error_handler.py",
            "main/server/mcp/utils/cache_manager.py"
        ]

    async def authenticate(self, username: str, password: str) -> Dict[str, str]:
        try:
            if not username or not password:
                raise MCPError(code=-32602, message="Username and password are required")
            checklist = await self.validate_checklist()
            if not checklist["all_files_present"]:
                raise MCPError(code=-32003, message="Missing required files", data={"checklist": checklist})
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

    async def validate_checklist(self) -> Dict[str, Any]:
        missing_files = []
        for file_path in self.required_files:
            if not Path(file_path).is_file():
                missing_files.append(file_path)
        fix_steps = [
            "1. Ensure all listed files are present in the main/server/mcp directory.",
            "2. Run 'git clone https://github.com/your-repo/vial-mcp.git' to restore missing files.",
            "3. Install dependencies with 'pip install -r requirements.txt' or 'npm install' as needed.",
            "4. Restart the server with 'docker-compose up -d'."
        ] if missing_files else []
        return {
            "all_files_present": len(missing_files) == 0,
            "missing_files": missing_files,
            "fix_steps": fix_steps
        }
