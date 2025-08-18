from config.config import DatabaseConfig
import jwt
import aiohttp
import logging
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.jwks_url = db.jwks_url
        self.jwt_audience = db.jwt_audience

    async def fetch_jwks(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jwks_url) as response:
                    return await response.json()
        except Exception as e:
            error_message = f"JWKS fetch failed: {str(e)} [security.py:20] [ID:jwks_fetch_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def verify_jwt(self, token: str) -> dict:
        try:
            jwks = await self.fetch_jwks()
            if "error" in jwks:
                return jwks
            key = next((k for k in jwks["keys"] if k["kid"] == "6_maVS8msnty"), None)
            if not key:
                error_message = "JWKS key not found [security.py:25] [ID:jwks_key_error]"
                logger.error(error_message)
                return {"error": error_message}
            public_key = jwt.algorithms.ECAlgorithm.from_jwk(key)
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["ES256"],
                audience=self.jwt_audience,
                options={"verify_exp": True}
            )
            logger.info(f"JWT verified for user: {decoded.get('sub')} [security.py:30] [ID:jwt_verify_success]")
            return decoded
        except Exception as e:
            error_message = f"JWT verification failed: {str(e)} [security.py:35] [ID:jwt_verify_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def log_action(self, user_id: str, action: str, data: dict):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, action, json.dumps(data), str(uuid.uuid4()), self.db.project_id]
            )
            logger.info(f"Action logged: {action} for user: {user_id} [security.py:40] [ID:log_action_success]")
        except Exception as e:
            logger.error(f"Action logging failed: {str(e)} [security.py:45] [ID:log_action_error]")

    async def log_error(self, user_id: str, action: str, error_message: str):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "error", json.dumps({"error": error_message}), str(uuid.uuid4()), self.db.project_id]
            )
            logger.info(f"Error logged: {error_message} for user: {user_id} [security.py:50] [ID:log_error_success]")
        except Exception as e:
            logger.error(f"Error logging failed: {str(e)} [security.py:55] [ID:log_error_error]")
