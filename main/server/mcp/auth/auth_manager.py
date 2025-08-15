# main/server/mcp/auth/auth_manager.py
import jwt
import datetime
import logging
import os
import re
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
        self.vials = ["vial1", "vial2", "vial3", "vial4"]  # Mock initial vials

    def sanitize_input(self, input_str: str) -> str:
        return re.sub(r'[<>&]', '', input_str)  # Basic sanitization

    async def authenticate(self, username: str, password: str) -> Dict[str, str]:
        try:
            username = self.sanitize_input(username)
            password = self.sanitize_input(password)
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
                vials_data = {vial: {"balance": 100.0} for vial in self.vials}  # Mock balances
                await self.cache.set_cache(f"auth:{username}", {"token": token, "vials": vials_data, "timestamp": datetime.datetime.utcnow().isoformat()})
                logger.info(f"User {username} authenticated with vials: {self.vials}")
                return {"access_token": token, "redirect": "/dashboard", "vials": vials_data}
            raise MCPError(code=-32001, message="Invalid credentials")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Authentication failed: {str(e)}")

    async def generate_api_key(self, user_id: str) -> str:
        try:
            api_key = jwt.encode({
                "user_id": user_id,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
                "scope": "api"
            }, SECRET_KEY, algorithm="HS256")
            await self.cache.set_cache(f"api:{user_id}", {"api_key": api_key, "timestamp": datetime.datetime.utcnow().isoformat()})
            logger.info(f"API key generated for {user_id}")
            return api_key
        except Exception as e:
            logger.error(f"API key generation error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"API key generation failed: {str(e)}")

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
            await self.cache.delete_cache(f"api:{user_id}")
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

    async def import_md(self, user_id: str, md_content: str) -> Dict[str, Any]:
        try:
            # Parse MD content and update vials (mock implementation)
            vials_data = {vial: {"balance": 100.0, "data": md_content} for vial in self.vials}
            await self.cache.set_cache(f"auth:{user_id}", {"vials": vials_data})
            return {"status": "imported", "vials": vials_data}
        except Exception as e:
            logger.error(f"Import error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Import failed: {str(e)}")

    async def export_md(self, user_id: str) -> str:
        try:
            cached = await self.cache.get_cache(f"auth:{user_id}")
            if not cached or "vials" not in cached:
                raise MCPError(code=-32004, message="No data to export")
            # Generate MD from vials (mock implementation)
            md_content = f"# Vial Data\n{' '.join([f'{k}: {v["balance"]}' for k, v in cached['vials'].items()])}"
            return md_content
        except Exception as e:
            logger.error(f"Export error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Export failed: {str(e)}")
