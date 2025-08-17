from config.config import ServerConfig
from lib.errors import ValidationError
from fastapi import HTTPException
import logging
import jwt
import aiohttp
from pydantic import BaseModel
from typing import Dict
import uuid

logger = logging.getLogger("mcp.auth")
logger.setLevel(logging.INFO)

class AuthInput(BaseModel):
    oauth_token: str
    provider: str

class AuthOutput(BaseModel):
    user_id: str
    jwt_token: str

class APICredentialInput(BaseModel):
    user_id: str

class APICredentialOutput(BaseModel):
    api_key: str
    api_secret: str

class AuthTool:
    def __init__(self, config: ServerConfig):
        self.config = config

    async def execute(self, input: Dict) -> Dict:
        try:
            if input.get("method") == "authentication":
                auth_input = AuthInput(**input)
                return await self.authenticate(auth_input)
            elif input.get("method") == "generateAPICredentials":
                cred_input = APICredentialInput(**input)
                return await self.generate_api_credentials(cred_input)
            else:
                raise ValidationError("Invalid auth method")
        except Exception as e:
            logger.error(f"Auth error: {str(e)}")
            raise HTTPException(400, str(e))

    async def authenticate(self, input: AuthInput) -> AuthOutput:
        try:
            if input.provider != "google":
                raise ValidationError("Only Google OAuth is supported")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://www.googleapis.com/oauth2/v3/tokeninfo?id_token={input.oauth_token}"
                ) as response:
                    if response.status != 200:
                        raise ValidationError("Invalid OAuth token")
                    data = await response.json()
                    if data.get("aud") != self.config.google_client_id:
                        raise ValidationError("Invalid client ID")
                    
                    user_id = data.get("sub")
                    if not user_id:
                        raise ValidationError("No user ID found in token")
                    
                    # Generate JWT
                    jwt_token = jwt.encode(
                        {"user_id": user_id},
                        self.config.jwt_secret,
                        algorithm="HS256"
                    )
                    
                    logger.info(f"Authenticated user: {user_id}")
                    return AuthOutput(user_id=user_id, jwt_token=jwt_token)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(400, str(e))

    async def generate_api_credentials(self, input: APICredentialInput) -> APICredentialOutput:
        try:
            # Verify user exists
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.x.ai/v1/verify_user?user_id={input.user_id}",
                    headers={"Authorization": f"Bearer {self.config.jwt_secret}"}
                ) as response:
                    if response.status != 200:
                        raise ValidationError("User verification failed")
            
            # Generate API key and secret
            api_key = str(uuid.uuid4())
            api_secret = str(uuid.uuid4())
            
            # Store credentials (simplified for demo; in production, store securely in DB)
            logger.info(f"Generated API credentials for user: {input.user_id}")
            return APICredentialOutput(api_key=api_key, api_secret=api_secret)
        except Exception as e:
            logger.error(f"API credential generation error: {str(e)}")
            raise HTTPException(400, str(e))
