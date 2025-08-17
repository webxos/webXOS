from config.config import ServerConfig
from lib.errors import ValidationError
from fastapi import HTTPException
import logging
import jwt
import aiohttp
from pydantic import BaseModel
from typing import Dict
import uuid
import hashlib

logger = logging.getLogger("mcp.auth")
logger.setLevel(logging.INFO)

class AuthInput(BaseModel):
    oauth_token: str
    provider: str

class AuthOutput(BaseModel):
    user_id: str
    jwt_token: str
    wallet_created: bool

class APICredentialInput(BaseModel):
    user_id: str

class APICredentialOutput(BaseModel):
    api_key: str
    api_secret: str

class AuthTool:
    def __init__(self, config: ServerConfig, db: DatabaseConfig):
        self.config = config
        self.db = db

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
                    
                    # Check if user exists
                    user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [user_id])
                    wallet_created = False
                    
                    if not user.rows:
                        # Create new wallet with four vials
                        wallet_address = str(uuid.uuid4())
                        api_key = str(uuid.uuid4())
                        api_secret = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
                        await self.db.query(
                            "INSERT INTO users (user_id, balance, wallet_address, api_key, api_secret, reputation) VALUES ($1, $2, $3, $4, $5, $6)",
                            [user_id, 0.0, wallet_address, api_key, api_secret, 0]
                        )
                        wallet_created = True
                        logger.info(f"Created new wallet for user: {user_id}")
                    
                    # Generate JWT
                    jwt_token = jwt.encode(
                        {"user_id": user_id},
                        self.config.jwt_secret,
                        algorithm="HS256"
                    )
                    
                    logger.info(f"Authenticated user: {user_id}")
                    return AuthOutput(user_id=user_id, jwt_token=jwt_token, wallet_created=wallet_created)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(400, str(e))

    async def generate_api_credentials(self, input: APICredentialInput) -> APICredentialOutput:
        try:
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            api_key = str(uuid.uuid4())
            api_secret = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
            
            await self.db.query(
                "UPDATE users SET api_key = $1, api_secret = $2 WHERE user_id = $3",
                [api_key, api_secret, input.user_id]
            )
            
            logger.info(f"Generated API credentials for user: {input.user_id}")
            return APICredentialOutput(api_key=api_key, api_secret=api_secret)
        except Exception as e:
            logger.error(f"API credential generation error: {str(e)}")
            raise HTTPException(400, str(e))
