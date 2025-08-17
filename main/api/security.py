from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from ..config.mcp_config import config
from ..utils.logging import log_error, log_info

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/oauth/token")

async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        client_id: str = payload.get("sub")
        if client_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        log_info(f"Token verified for client_id: {client_id}")
        return token
    except JWTError as e:
        log_error(f"Traceback: Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token error: {str(e)}")
