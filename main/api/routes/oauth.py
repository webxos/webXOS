from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ...security.authentication import create_access_token, verify_credentials
from pymongo import MongoClient
from ...config.settings import settings
import uuid

router = APIRouter()

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str

class CredentialsRequest(BaseModel):
    user_id: str

@router.post("/oauth/token")
async def get_token(request: TokenRequest):
    if request.grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="Invalid grant_type")
    user = verify_credentials(request.client_id, request.client_secret)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token({"sub": user["user_id"]})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/generate-credentials")
async def generate_credentials(request: CredentialsRequest):
    try:
        client = MongoClient(settings.database.url)
        db = client[settings.database.db_name]
        api_key = f"WEBXOS-{uuid.uuid4().hex[:9]}"
        api_secret = uuid.uuid4().hex
        db.credentials.insert_one({
            "user_id": request.user_id,
            "api_key": api_key,
            "api_secret": api_secret
        })
        client.close()
        return {"apiKey": api_key, "apiSecret": api_secret}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/authenticate")
async def authenticate(token: str = Depends(verify_token)):
    return {"message": "Authenticated"}
