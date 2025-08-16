from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ...security.authentication import verify_token

auth_scheme = HTTPBearer()

async def auth_middleware(request: Request, call_next):
    if request.url.path in ["/v1/oauth/token", "/v1/generate-credentials"]:
        return await call_next(request)
    credentials: HTTPAuthorizationCredentials = await auth_scheme(request)
    token = verify_token(credentials.credentials)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    request.state.user = token
    return await call_next(request)
