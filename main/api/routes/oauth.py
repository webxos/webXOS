from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...config.mcp_config import config
from ...utils.logging import log_error, log_info
from jose import jwt
import datetime

router = APIRouter()

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

@router.post("/oauth/token", response_model=TokenResponse)
async def issue_token(token_request: TokenRequest):
    try:
        if token_request.grant_type != "client_credentials":
            raise HTTPException(status_code=400, detail="Invalid grant_type")
        if token_request.client_id != "WEBXOS-MOCKKEY" or token_request.client_secret != "MOCKSECRET1234567890":
            raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        payload = {
            "sub": token_request.client_id,
            "iat": datetime.datetime.utcnow(),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }
        token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)
        log_info(f"Token issued for client_id: {token_request.client_id}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600
        }
    except Exception as e:
        log_error(f"Traceback: OAuth token issuance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OAuth error: {str(e)}")
