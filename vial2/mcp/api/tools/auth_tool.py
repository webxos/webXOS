import aiohttp
import jwt
from config.config import DatabaseConfig
from lib.security import SecurityHandler
import logging
import uuid

logger = logging.getLogger(__name__)

class AuthTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.client_id = db.stack_auth_client_id
        self.client_secret = db.stack_auth_client_secret
        self.audience = db.jwt_audience
        self.redirect_uri = "https://webxos.netlify.app/vial2.html"
        self.auth_url = "https://stack-auth.com/oauth/authorize"
        self.token_url = "https://stack-auth.com/oauth/token"
        self.jwks_url = "https://stack-auth.com/.well-known/jwks.json"

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            if method == "oauth_login":
                return await self.oauth_login(data.get("code"), data.get("code_verifier"), data.get("project_id"))
            elif method == "generate_api_key":
                return await self.generate_api_key(data.get("user_id"), data.get("project_id"))
            else:
                error_message = f"Invalid auth method: {method} [auth_tool.py:25] [ID:auth_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Auth operation failed: {str(e)} [auth_tool.py:30] [ID:auth_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def oauth_login(self, code: str, code_verifier: str, project_id: str) -> dict:
        try:
            if project_id != self.db.project_id:
                error_message = f"Invalid project ID: {project_id} [auth_tool.py:35] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    data={
                        "grant_type": "authorization_code",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "code_verifier": code_verifier,
                        "redirect_uri": self.redirect_uri
                    }
                ) as response:
                    token_data = await response.json()
                    if "error" in token_data:
                        error_message = f"OAuth token error: {token_data['error']} [auth_tool.py:45] [ID:oauth_error]"
                        logger.error(error_message)
                        return {"error": error_message}
                    access_token = token_data["access_token"]
                    decoded = jwt.decode(
                        access_token,
                        algorithms=["RS256"],
                        audience=self.audience,
                        options={"verify_signature": False}
                    )
                    user_id = decoded.get("sub")
                    await self.db.query(
                        "INSERT INTO sessions (session_id, user_id, access_token, expires_at, project_id) VALUES ($1, $2, $3, $4, $5)",
                        [str(uuid.uuid4()), user_id, access_token, token_data.get("expires_at"), project_id]
                    )
                    await self.security.log_action(user_id, "login", {"method": "oauth"})
                    logger.info(f"OAuth login successful for user {user_id} [auth_tool.py:55] [ID:oauth_success]")
                    return {"access_token": access_token, "user_id": user_id}
        except Exception as e:
            error_message = f"OAuth login failed: {str(e)} [auth_tool.py:60] [ID:oauth_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def generate_api_key(self, user_id: str, project_id: str) -> dict:
        try:
            if project_id != self.db.project_id:
                error_message = f"Invalid project ID: {project_id} [auth_tool.py:65] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            api_key = str(uuid.uuid4())
            await self.db.query(
                "INSERT INTO api_keys (api_key, user_id, project_id) VALUES ($1, $2, $3)",
                [api_key, user_id, project_id]
            )
            await self.security.log_action(user_id, "generate_api_key", {"api_key": api_key[:8] + "..."})
            logger.info(f"API key generated for user {user_id} [auth_tool.py:70] [ID:api_key_success]")
            return {"api_key": api_key}
        except Exception as e:
            error_message = f"API key generation failed: {str(e)} [auth_tool.py:75] [ID:api_key_error]"
            logger.error(error_message)
            return {"error": error_message}
