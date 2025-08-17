import jwt
import aiohttp
from config.config import DatabaseConfig
import logging
import uuid

logger = logging.getLogger(__name__)

class SecurityHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.jwks_url = "https://stack-auth.com/.well-known/jwks.json"
        self.audience = db.jwt_audience
        self.jwks = None

    async def fetch_jwks(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jwks_url) as response:
                    self.jwks = await response.json()
                    logger.info("JWKS fetched successfully [security.py:20] [ID:jwks_success]")
        except Exception as e:
            logger.error(f"JWKS fetch failed: {str(e)} [security.py:25] [ID:jwks_error]")
            raise

    async def verify_jwt(self, token: str) -> dict:
        try:
            if not self.jwks:
                await self.fetch_jwks()
            headers = jwt.get_unverified_header(token)
            kid = headers["kid"]
            key = next((k for k in self.jwks["keys"] if k["kid"] == kid), None)
            if not key:
                error_message = f"JWKS key not found: {kid} [security.py:30] [ID:jwks_key_error]"
                logger.error(error_message)
                return {"error": error_message}
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.audience
            )
            logger.info(f"JWT verified for user {decoded.get('sub')} [security.py:35] [ID:jwt_success]")
            return decoded
        except Exception as e:
            error_message = f"JWT verification failed: {str(e)} [security.py:40] [ID:jwt_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def log_action(self, user_id: str, action: str, data: dict):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, action, json.dumps(data), str(uuid.uuid4()), self.db.project_id]
            )
            logger.info(f"Action logged for user {user_id}: {action} [security.py:45] [ID:action_log_success]")
        except Exception as e:
            logger.error(f"Action logging failed: {str(e)} [security.py:50] [ID:action_log_error]")
            raise

    async def log_error(self, user_id: str, action: str, error_message: str):
        try:
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, f"error_{action}", json.dumps({"error": error_message}), str(uuid.uuid4()), self.db.project_id]
            )
            logger.info(f"Error logged for user {user_id}: {error_message} [security.py:55] [ID:error_log_success]")
        except Exception as e:
            logger.error(f"Error logging failed: {str(e)} [security.py:60] [ID:error_log_error]")
            raise
