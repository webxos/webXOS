from fastapi import Depends, HTTPException
from jose import JWTError, jwt
from ...config.mcp_config import mcp_config
from ...utils.logging import log_error, log_info

async def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, mcp_config.JWT_SECRET_KEY, algorithms=[mcp_config.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            log_error("Invalid JWT token: no user_id")
            raise HTTPException(status_code=401, detail="Invalid token")
        log_info(f"Token verified for user {user_id}")
        return user_id
    except JWTError as e:
        log_error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")
