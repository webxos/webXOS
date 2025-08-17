from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from ...utils.logging import log_error, log_info
from ...config.mcp_config import mcp_config
from pydantic import BaseModel

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/oauth/token")

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

@router.post("/oauth/token")
async def token(request: TokenRequest):
    try:
        if request.grant_type != "client_credentials":
            raise ValueError("Unsupported grant type")
        if request.client_id != "WEBXOS-MOCKKEY" or request.client_secret != "MOCKSECRET1234567890":
            raise ValueError("Invalid client credentials")
        payload = {"sub": request.client_id, "exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(payload, mcp_config.JWT_SECRET_KEY, algorithm=mcp_config.ALGORITHM)
        log_info(f"Token issued for {request.client_id}")
        return TokenResponse(access_token=token, token_type="bearer")
    except Exception as e:
        log_error(f"OAuth token error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")
