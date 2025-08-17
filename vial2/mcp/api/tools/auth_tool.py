from config.config import DatabaseConfig
from lib.security import SecurityHandler
import aiohttp
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class AuthTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            user_id = data.get("user_id")
            project_id = data.get("project_id", self.db.project_id)
            if project_id != self.db.project_id:
                error_message = f"Invalid project ID: {project_id} [auth_tool.py:20] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            if method == "authenticate":
                return await self.authenticate(data.get("code"), data.get("redirect_uri"))
            elif method == "validate_wallet":
                return await self.validate_wallet(user_id, data.get("wallet_data"))
            else:
                error_message = f"Invalid auth method: {method} [auth_tool.py:25] [ID:auth_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Auth operation failed: {str(e)} [auth_tool.py:30] [ID:auth_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def authenticate(self, code: str, redirect_uri: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                token_response = await session.post(
                    "https://api.stack-auth.com/api/v1/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "client_id": self.db.stack_auth_client_id,
                        "client_secret": self.db.stack_auth_client_secret,
                        "code": code,
                        "redirect_uri": redirect_uri
                    }
                )
                token_data = await token_response.json()
                if "access_token" not in token_data:
                    error_message = f"Token exchange failed: {token_data.get('error', 'Unknown')} [auth_tool.py:35] [ID:token_error]"
                    logger.error(error_message)
                    return {"error": error_message}
                access_token = token_data["access_token"]
                decoded = await self.security.verify_jwt(access_token)
                if "error" in decoded:
                    error_message = decoded["error"]
                    logger.error(error_message)
                    return {"error": error_message}
                user_id = decoded.get("sub")
                await self.db.query(
                    "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                    [str(uuid.uuid4()), user_id, "auth", json.dumps({"access_token": access_token}), str(uuid.uuid4()), self.db.project_id]
                )
                logger.info(f"Authentication successful for user: {user_id} [auth_tool.py:40] [ID:auth_success]")
                return {"status": "success", "access_token": access_token, "user_id": user_id}
        except Exception as e:
            error_message = f"Authentication failed: {str(e)} [auth_tool.py:45] [ID:auth_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def validate_wallet(self, user_id: str, wallet_data: dict) -> dict:
        try:
            if not wallet_data.get("address") or not wallet_data.get("signature"):
                error_message = "Invalid wallet data [auth_tool.py:50] [ID:wallet_validation_error]"
                logger.error(error_message)
                return {"error": error_message}
            # Simulate .md wallet validation (replace with actual logic)
            is_valid = wallet_data["address"].startswith("0x")
            if not is_valid:
                error_message = f"Invalid wallet address: {wallet_data['address']} [auth_tool.py:55] [ID:wallet_invalid_error]"
                logger.error(error_message)
                return {"error": error_message}
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "wallet_validation", json.dumps(wallet_data), str(uuid.uuid4()), self.db.project_id]
            )
            logger.info(f"Wallet validated for user: {user_id} [auth_tool.py:60] [ID:wallet_validation_success]")
            return {"status": "success", "wallet_address": wallet_data["address"]}
        except Exception as e:
            error_message = f"Wallet validation failed: {str(e)} [auth_tool.py:65] [ID:wallet_validation_error]"
            logger.error(error_message)
            return {"error": error_message}
