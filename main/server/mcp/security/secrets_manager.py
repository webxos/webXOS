# main/server/mcp/security/secrets_manager.py
from typing import Optional
from pymongo import MongoClient
import redis.asyncio as redis
from ..utils.mcp_error_handler import MCPError
import os
import secrets
import base64

class SecretsManager:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.collection = self.db["secrets"]
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.default_ttl = 3600  # 1 hour

    async def store_secret(self, user_id: str, secret_name: str, secret_value: str) -> str:
        try:
            if not secret_name or not secret_value:
                raise MCPError(code=-32602, message="Secret name and value are required")
            secret_id = secrets.token_hex(16)
            encrypted_value = base64.b64encode(secret_value.encode()).decode()  # Simplified encryption
            secret = {
                "secret_id": secret_id,
                "user_id": user_id,
                "secret_name": secret_name,
                "secret_value": encrypted_value
            }
            self.collection.insert_one(secret)
            await self.redis_client.setex(f"secret:{secret_id}", self.default_ttl, encrypted_value)
            return secret_id
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to store secret: {str(e)}")

    async def retrieve_secret(self, user_id: str, secret_id: str) -> Optional[str]:
        try:
            # Check Redis cache first
            cached_value = await self.redis_client.get(f"secret:{secret_id}")
            if cached_value:
                return base64.b64decode(cached_value).decode()

            # Fallback to MongoDB
            secret = self.collection.find_one({"secret_id": secret_id, "user_id": user_id})
            if not secret:
                raise MCPError(code=-32003, message="Secret not found or access denied")
            decrypted_value = base64.b64decode(secret["secret_value"]).decode()
            await self.redis_client.setex(f"secret:{secret_id}", self.default_ttl, secret["secret_value"])
            return decrypted_value
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to retrieve secret: {str(e)}")

    async def delete_secret(self, user_id: str, secret_id: str) -> None:
        try:
            result = self.collection.delete_one({"secret_id": secret_id, "user_id": user_id})
            if result.deleted_count == 0:
                raise MCPError(code=-32003, message="Secret not found or access denied")
            await self.redis_client.delete(f"secret:{secret_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to delete secret: {str(e)}")

    def close(self):
        self.mongo_client.close()
