import jwt
import aiohttp
import logging
from config.config import DatabaseConfig
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.jwks_url = "https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/.well-known/jwks.json"
        self.project_id = "twilight-art-21036984"

    async def log_action(self, user_id: str, action: str, data: dict):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "action", json.dumps({"action": action, "data": data}), str(uuid.uuid4()), self.project_id]
            )
            logger.info(f"Action logged for user {user_id}: {action}")
        except Exception as e:
            logger.error(f"Action logging failed: {str(e)}")

    async def log_error(self, user_id: str, action: str, error: str):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "error", json.dumps({"action": action, "error": error}), str(uuid.uuid4()), self.project_id]
            )
            logger.error(f"Error logged for user {user_id}: {error}")
        except Exception as e:
            logger.error(f"Error logging failed: {str(e)}")

    async def verify_jwt(self, token: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jwks_url) as resp:
                    jwks = await resp.json()
            keys = jwks.get("keys", [])
            for key in keys:
                try:
                    return jwt.decode(token, key, algorithms=["RS256"])
                except jwt.InvalidTokenError:
                    continue
            raise ValueError("Invalid JWT token")
        except Exception as e:
            logger.error(f"JWT verification failed: {str(e)}")
            raise ValueError(f"JWT verification failed: {str(e)}")
