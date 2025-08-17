from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
from ..config.mcp_config import config
from ..utils.logging import log_error, log_info

router = APIRouter()

class OAuthRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

@router.post("/oauth/token")
async def oauth_token(request: OAuthRequest):
    try:
        if request.grant_type != "client_credentials":
            raise HTTPException(status_code=400, detail="Invalid grant type")
        if request.client_id != "WEBXOS-MOCKKEY" or request.client_secret != "MOCKSECRET1234567890":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = jwt.encode(
            {
                "sub": request.client_id,
                "exp": datetime.utcnow() + timedelta(hours=1),
                "iat": datetime.utcnow()
            },
            config.SECRET_KEY,
            algorithm=config.ALGORITHM
        )
        log_info(f"OAuth token generated for client_id: {request.client_id}")
        return {"access_token": token, "token_type": "bearer", "expires_in": 3600}
    except Exception as e:
        log_error(f"Traceback: OAuth token generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
