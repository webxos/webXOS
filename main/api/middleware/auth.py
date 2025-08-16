from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ...security.authentication import verify_token
from ...utils.logging import log_error

security = HTTPBearer()

async def auth_middleware(request: Request, call_next):
    """JWT authentication middleware."""
    try:
        if request.url.path.startswith("/v1/oauth") or request.url.path == "/health":
            return await call_next(request)
        
        credentials: HTTPAuthorizationCredentials = await security(request)
        user_id = await verify_token(credentials)
        request.state.user_id = user_id
        response = await call_next(request)
        return response
    except HTTPException as e:
        log_error(f"Auth middleware failed: {str(e)}")
        raise
    except Exception as e:
        log_error(f"Auth middleware error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
