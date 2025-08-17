from config.config import DatabaseConfig, APIConfig
from lib.errors import ValidationError
from lib.security import SecurityHandler
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
import logging
import hashlib
import uuid
import httpx
import base64
import os
import secrets

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
    code_verifier: str

class AuthTokenOutput(BaseModel):
    access_token: str
    user_id: str
    session_id: str

class AuthTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.api_config = APIConfig()
        self.security_handler = SecurityHandler(db)
        self.redirect_uri_allowlist = [os.getenv("OAUTH_REDIRECT_URI", "https://webxos.netlify.app/auth/callback")]

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
            await self.security_handler.log_event(
                event_type="auth_error",
                user_id=input.get("user_id"),
                details={"error": str(e)}
            )
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
            
            await self.security_handler.log_event(
                event_type="api_credentials_generated",
                user_id=input.user_id,
                details={"api_key": api_key}
            )
            logger.info(f"Generated API credentials for {input.user_id}")
            return AuthGenerateOutput(api_key=api_key, api_secret=api_secret)
        except Exception as e:
            logger.error(f"Generate API credentials error: {str(e)}")
            await self.security_handler.log_event(
                event_type="api_credentials_error",
                user_id=input.user_id,
                details={"error": str(e)}
            )
            raise HTTPException(400, str(e))

    async def exchange_token(self, input: AuthTokenInput) -> AuthTokenOutput:
        try:
            if input.redirect_uri not in self.redirect_uri_allowlist:
                raise ValidationError(f"Invalid redirect URI: {input.redirect_uri}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": self.api_config.github_client_id,
                        "client_secret": self.api_config.github_client_secret,
                        "code": input.code,
                        "redirect_uri": input.redirect_uri,
                        "code_verifier": input.code_verifier
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
                
                # Validate token audience (ensure issued to this app)
                if user_data.get("aud") != self.api_config.github_client_id:
                    raise ValidationError("Invalid token audience")
                
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
                
                # Create secure session
                session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
                expires_at = datetime.utcnow() + timedelta(minutes=15)
                await self.db.query(
                    "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
                    [session_id, user_id, expires_at]
                )
                
                await self.db.query(
                    "UPDATE users SET access_token = $1 WHERE user_id = $2",
                    [access_token, user_id]
                )
                
                await self.security_handler.log_event(
                    event_type="auth_success",
                    user_id=user_id,
                    details={"access_token": access_token[:8] + "...", "session_id": session_id}
                )
                logger.info(f"Exchanged OAuth token for user {user_id}")
                return AuthTokenOutput(access_token=access_token, user_id=user_id, session_id=session_id)
        except Exception as e:
            logger.error(f"Exchange token error: {str(e)}")
            await self.security_handler.log_event(
                event_type="auth_error",
                user_id=user_id,
                details={"error": str(e), "redirect_uri": input.redirect_uri}
            )
            raise HTTPException(400, str(e))

    async def verify_token(self, token: str, session_id: str) -> Dict[str, Any]:
        try:
            # Verify session
            session = await self.db.query(
                "SELECT session_key, user_id, expires_at FROM sessions WHERE session_key = $1",
                [session_id]
            )
            if not session.rows or session.rows[0]["expires_at"] < datetime.utcnow():
                return None
            
            user = await self.db.query(
                "SELECT user_id, access_token FROM users WHERE user_id = $1 AND access_token = $2",
                [session.rows[0]["user_id"], token]
            )
            if not user.rows:
                return None
            
            await self.security_handler.log_event(
                event_type="token_verified",
                user_id=user.rows[0]["user_id"],
                details={"session_id": session_id}
            )
            return {"user_id": user.rows[0]["user_id"]}
        except Exception as e:
            logger.error(f"Verify token error: {str(e)}")
            await self.security_handler.log_event(
                event_type="token_verification_error",
                user_id=None,
                details={"error": str(e), "session_id": session_id}
            )
            return None
