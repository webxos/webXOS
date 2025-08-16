from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from ...config.settings import settings
from ...utils.logging import log_error

security = HTTPBearer()

def create_jwt_token(payload: dict) -> str:
    """Create JWT token."""
    try:
        token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
        return token
    except Exception as e:
        log_error(f"JWT creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user_id."""
    try:
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            log_error("Invalid JWT: Missing sub claim")
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError as e:
        log_error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))
