from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.auth import generate_jwt_token

router = APIRouter()

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

@router.post("/oauth/token")
async def token(request: TokenRequest):
    if request.grant_type != "client_credentials" or request.client_id != "WEBXOS-MOCKKEY" or request.client_secret != "MOCKSECRET1234567890":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = generate_jwt_token({"sub": request.client_id})
    return {"access_token": token, "token_type": "bearer"}
