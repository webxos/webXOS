import uuid
import aiohttp
import jwt
import asyncpg
from config.config import DatabaseConfig
from postgrest import AsyncPostgrestClient
import logging

logger = logging.getLogger(__name__)

class AuthTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.jwks_url = "https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/.well-known/jwks.json"
        self.client_id = "your_stack_auth_client_id"
        self.client_secret = "your_stack_auth_client_secret"
        self.data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")
        self.project_id = "twilight-art-21036984"

    async def execute(self, args: dict) -> dict:
        method = args.get("method")
        if method == "oauth_login":
            return await self.oauth_login(args.get("code"), args.get("code_verifier"), args.get("project_id"))
        elif method == "generateAPICredentials":
            return await self.generate_api_credentials(args.get("user_id"))
        else:
            raise ValueError("Unknown auth method")

    async def oauth_login(self, code: str, code_verifier: str, project_id: str) -> dict:
        if project_id != self.project_id:
            raise ValueError("Invalid Neon project ID")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "code_verifier": code_verifier,
                    "redirect_uri": "https://webxos.netlify.app/vial2/callback"
                }
            ) as resp:
                if resp.status != 200:
                    logger.error(f"OAuth token request failed: {resp.status}")
                    raise ValueError("OAuth token request failed")
                token_data = await resp.json()
                access_token = token_data["access_token"]
                user_info = jwt.decode(access_token, options={"verify_signature": False})
                user_id = user_info.get("sub")
                await self.db.query(
                    "INSERT INTO users (user_id, github_id, email, project_id) VALUES ($1, $2, $3, $4) ON CONFLICT (user_id) DO NOTHING",
                    [user_id, user_info.get("github_id", ""), user_info.get("email", ""), project_id]
                )
                self.data_api.auth(access_token)
                await self.data_api.from_("users").insert({"user_id": user_id, "email": user_info.get("email", ""), "project_id": project_id}).eq("user_id", user_id).execute()
                await self.db.query(
                    "INSERT INTO sessions (session_id, user_id, access_token, expires_at, project_id) VALUES ($1, $2, $3, CURRENT_TIMESTAMP + INTERVAL '1 hour', $4)",
                    [str(uuid.uuid4()), user_id, access_token, project_id]
                )
                await self.db.query(
                    "INSERT INTO wallets (wallet_id, user_id, address, balance, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (user_id) DO NOTHING",
                    [str(uuid.uuid4()), user_id, f"webxos_{user_id}", 0.0, str(uuid.uuid4()), project_id]
                )
                logger.info(f"User {user_id} authenticated successfully")
                return {"access_token": access_token, "user_id": user_id}
