from config.config import DatabaseConfig, APIConfig
from lib.errors import ValidationError
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
import logging
import hashlib
import uuid
import httpx
import os

logger = logging.getLogger("mcp.auth")
logger.setLevel(logging.INFO)

class AuthGenerateInput(BaseModel):
    user_id: str

class AuthGenerateOutput(BaseModel):
    api_key: str
    api_secret: str

class AuthTokenInput(BaseModel):
    code: str
    redirect_uri: str

class AuthTokenOutput(BaseModel):
    access_token: str
    user_id: str

class AuthTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.api_config = APIConfig()

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "generateAPICredentials")
            if method == "generateAPICredentials":
                generate_input = AuthGenerateInput(**input)
                return await self.generate_api_credentials(generate_input)
            elif method == "exchangeToken":
                token_input = AuthTokenInput(**input)
                return await self.exchange_token(token_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Auth error: {str(e)}")
            raise HTTPException(400, str(e))

    async def generate_api_credentials(self, input: AuthGenerateInput) -> AuthGenerateOutput:
        try:
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            api_key = str(uuid.uuid4())
            api_secret = str(uuid.uuid4())
            api_secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()
            
            await self.db.query(
                "UPDATE users SET api_key = $1, api_secret = $2 WHERE user_id = $3",
                [api_key, api_secret_hash, input.user_id]
            )
            
            logger.info(f"Generated API credentials for {input.user_id}")
            return AuthGenerateOutput(api_key=api_key, api_secret=api_secret)
        except Exception as e:
            logger.error(f"Generate API credentials error: {str(e)}")
            raise HTTPException(400, str(e))

    async def exchange_token(self, input: AuthTokenInput) -> AuthTokenOutput:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": self.api_config.github_client_id,
                        "client_secret": self.api_config.github_client_secret,
                        "code": input.code,
                        "redirect_uri": input.redirect_uri
                    },
                    headers={"Accept": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    raise ValidationError(f"GitHub OAuth error: {data['error_description']}")
                
                access_token = data["access_token"]
                user_response = await client.get(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                user_response.raise_for_status()
                user_data = user_response.json()
                user_id = str(user_data["id"])
                
                existing_user = await self.db.query(
                    "SELECT user_id FROM users WHERE user_id = $1",
                    [user_id]
                )
                if not existing_user.rows:
                    wallet_address = str(uuid.uuid4())
                    await self.db.query(
                        "INSERT INTO users (user_id, balance, wallet_address, access_token) VALUES ($1, $2, $3, $4)",
                        [user_id, 0.0, wallet_address, access_token]
                    )
                    from tools.wallet import WalletTool
                    wallet_tool = WalletTool(self.db)
                    await wallet_tool.initialize_new_wallet(user_id, wallet_address, str(uuid.uuid4()), str(uuid.uuid4()))
                
                await self.db.query(
                    "UPDATE users SET access_token = $1 WHERE user_id = $2",
                    [access_token, user_id]
                )
                
                logger.info(f"Exchanged OAuth token for user {user_id}")
                return AuthTokenOutput(access_token=access_token, user_id=user_id)
        except Exception as e:
            logger.error(f"Exchange token error: {str(e)}")
            raise HTTPException(400, str(e))

    async def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            user = await self.db.query(
                "SELECT user_id, access_token FROM users WHERE access_token = $1",
                [token]
            )
            if not user.rows:
                return None
            return {"user_id": user.rows[0]["user_id"]}
        except Exception as e:
            logger.error(f"Verify token error: {str(e)}")
            return None
