# main/server/mcp/auth/auth_manager.py
import jwt
import datetime
import logging
from typing import Tuple, Dict
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")

async def authenticate_user(username: str, password: str) -> Tuple[str, Dict]:
    try:
        if username != "test_user" or password != "test_pass":
            raise MCPError(code=-32001, message="Invalid credentials")
        token = jwt.encode({
            "sub": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, "SECRET_KEY", algorithm="HS256")
        vials = {f"vial{i}": {"balance": 100 * i} for i in range(1, 5)}
        logger.info(f"User {username} authenticated")
        return token, vials
    except MCPError as e:
        raise e
    except Exception as e:
        logger.error(f"Auth error: {str(e)}", exc_info=True)
        raise MCPError(code=-32603, message=f"Authentication failed: {str(e)}")
