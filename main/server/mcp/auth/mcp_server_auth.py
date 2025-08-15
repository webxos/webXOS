# main/server/mcp/auth/mcp_server_auth.py
import jwt
import datetime
from fastapi import HTTPException
from typing import Dict, Tuple
from ..utils.mcp_error_handler import MCPError

async def authenticate_user(username: str, password: str) -> Tuple[str, Dict]:
    if username != "test_user" or password != "test_pass":
        raise MCPError(code=-32001, message="Invalid credentials")
    token = jwt.encode({
        "sub": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, "SECRET_KEY", algorithm="HS256")
    vials = {f"vial{i}": {"balance": 100 * i} for i in range(1, 5)}
    return token, vials

async def authenticate_oauth(provider: str, code: str) -> Dict[str, Any]:
    try:
        if provider != "mock" or code != "test_code":
            raise MCPError(code=-32002, message="Invalid OAuth provider or code")
        token = jwt.encode({
            "sub": "oauth_user",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, "SECRET_KEY", algorithm="HS256")
        return {"access_token": token, "vials": {"vial1": {"balance": 200}}}
    except MCPError as e:
        raise e
    except Exception as e:
        raise MCPError(code=-32603, message=f"OAuth failed: {str(e)}")
