from config.config import DatabaseConfig
import logging
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
from pydantic import BaseModel
from fastapi import HTTPException
import os

logger = logging.getLogger("mcp.auth")
logger.setLevel(logging.INFO)

class AuthInput(BaseModel):
    oauth_token: str
    provider: str

class AuthOutput(BaseModel):
    access_token: str
    expires_in: int
    user_id: str

class AuthenticationTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not self.client_id:
            raise ValueError("GOOGLE_CLIENT_ID not set")

    async def execute(self, input: dict) -> AuthOutput:
        try:
            auth_input = AuthInput(**input)
            if auth_input.provider != "google":
                raise HTTPException(400, "Unsupported OAuth provider")

            # Verify Google OAuth token
            idinfo = id_token.verify_oauth2_token(
                auth_input.oauth_token, requests.Request(), self.client_id
            )

            if not idinfo.get("sub") or not idinfo.get("email"):
                raise HTTPException(400, "Invalid OAuth token")

            user_id = f"user_{idinfo['sub']}"
            email = idinfo["email"]
            username = idinfo.get("name", email.split("@")[0])

            # Check if user exists, create if not
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [user_id])
            if not user.rows:
                await self.db.query(
                    """
                    INSERT INTO users (user_id, username, wallet_address, balance, reputation, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    [user_id, username, f"wallet_{user_id}", 0, 0, "now()"]
                )

            # Generate JWT
            jwt_secret = os.getenv("JWT_SECRET", "default_secret")
            access_token = jwt.encode(
                {"user_id": user_id}, jwt_secret, algorithm="HS256"
            )
            logger.info(f"User authenticated: {user_id} via {auth_input.provider}")
            return AuthOutput(
                access_token=access_token,
                expires_in=86400,
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(400, str(e))
