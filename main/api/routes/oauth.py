from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...security.authentication import create_jwt_token
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis
import json

router = APIRouter(prefix="/v1/oauth", tags=["OAuth"])

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

@router.post("/token", response_model=TokenResponse)
async def issue_token(request: TokenRequest, redis=Depends(get_redis)):
    """Issue JWT access token."""
    try:
        if request.grant_type != "client_credentials":
            log_error(f"Invalid grant type: {request.grant_type}")
            raise HTTPException(status_code=400, detail="Invalid grant type")
        
        # Mock credential validation (replace with MongoDB query)
        if not (request.client_id.startswith("WEBXOS-") and len(request.client_secret) > 10):
            log_error(f"Invalid credentials for client_id: {request.client_id}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_jwt_token({"sub": request.client_id})
        await redis.set(f"token:{request.client_id}", token, ex=3600)
        log_info(f"Token issued for client {request.client_id}")
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=3600
        )
    except Exception as e:
        log_error(f"Token issuance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-credentials")
async def generate_credentials(user_id: str = Depends(verify_token)):
    """Generate new API credentials."""
    try:
        api_key = f"WEBXOS-{np.random.bytes(8).hex()}"
        api_secret = np.random.bytes(16).hex()
        # Mock credential storage (replace with MongoDB insert)
        log_info(f"Credentials generated for user {user_id}")
        return {"apiKey": api_key, "apiSecret": api_secret}
    except Exception as e:
        log_error(f"Credential generation failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
