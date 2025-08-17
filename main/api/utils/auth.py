from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import jwt
from main.api.config.mcp_config import MCP_CONFIG

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/oauth/token")

def generate_jwt_token(payload: dict) -> str:
    return jwt.encode(payload, MCP_CONFIG["JWT_SECRET"], algorithm="HS256")

async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, MCP_CONFIG["JWT_SECRET"], algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
