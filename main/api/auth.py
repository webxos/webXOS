from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
from ..utils.logging import log_error, log_info

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/oauth/token")
SECRET_KEY = "your_jwt_secret_here"
ALGORITHM = "HS256"

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
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        log_info(f"OAuth token generated for client_id: {request.client_id}")
        return {"access_token": token, "token_type": "bearer", "expires_in": 3600}
    except Exception as e:
        log_error(f"Traceback: OAuth token generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
