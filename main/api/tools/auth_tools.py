from config.config import DatabaseConfig
from lib.errors import ValidationError
from lib.security import Security
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
import hashlib
import time

logger = logging.getLogger("mcp.auth")
logger.setLevel(logging.INFO)

class AuthenticationInput(BaseModel):
    oauth_token: str
    provider: str

class AuthenticationOutput(BaseModel):
    user_id: str
    access_token: str
    expires_in: int

class AuthenticationTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = Security(db)

    async def execute(self, input: Dict[str, Any]) -> AuthenticationOutput:
        try:
            auth_input = AuthenticationInput(**input)
            if auth_input.provider != "google":
                raise ValidationError("Unsupported OAuth provider")
            
            # Verify Google OAuth token
            user_info = id_token.verify_oauth2_token(
                auth_input.oauth_token,
                requests.Request(),
                self.db.google_client_id
            )
            
            user_id = f"user_{user_info['sub']}"
            username = user_info.get("name", user_info.get("email", "unknown"))
            
            # Check if user exists
            user = await self.db.query(
                "SELECT user_id, wallet_address FROM users WHERE user_id = $1",
                [user_id]
            )
            
            if not user.rows:
                # Create new user with auto-generated wallet
                wallet_address = hashlib.sha256(
                    f"{user_id}{int(time.time())}".encode()
                ).hexdigest()
                await self.db.query(
                    "INSERT INTO users (user_id, username, wallet_address, balance, reputation) VALUES ($1, $2, $3, $4, $5)",
                    [user_id, username, wallet_address, 0.0, 0]
                )
                logger.info(f"Created new user {user_id} with wallet {wallet_address}")
            else:
                wallet_address = user.rows[0]["wallet_address"]
            
            # Generate JWT
            expires_in = 86400  # 24 hours
            access_token = jwt.encode(
                {"sub": user_id, "exp": int(time.time()) + expires_in},
                self.db.jwt_secret,
                algorithm="HS256"
            )
            
            # Store session
            await self.db.query(
                "INSERT INTO sessions (user_id, access_token, expires_at) VALUES ($1, $2, $3)",
                [user_id, access_token, f"now() + interval '{expires_in} seconds'"]
            )
            
            logger.info(f"Authenticated user: {user_id}")
            return AuthenticationOutput(
                user_id=user_id,
                access_token=access_token,
                expires_in=expires_in
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(400, str(e))
