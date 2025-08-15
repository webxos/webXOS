# main/server/mcp/security/secrets_manager.py
from typing import Dict, Any, Optional
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
import os
import logging
import secrets
import base64
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

logger = logging.getLogger("mcp")

class SecretsManager:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.secrets = self.db["secrets"]
        self.key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        self.cipher = Fernet(self.key)
        self.rotation_interval = timedelta(days=30)

    async def store_secret(self, user_id: str, secret_name: str, secret_value: str) -> str:
        try:
            if not user_id or not secret_name or not secret_value:
                raise MCPError(code=-32602, message="User ID, secret name, and value are required")
            
            secret_id = secrets.token_hex(16)
            encrypted_value = self.cipher.encrypt(secret_value.encode()).decode()
            
            secret_record = {
                "secret_id": secret_id,
                "user_id": user_id,
                "secret_name": secret_name,
                "secret_value": encrypted_value,
                "created_at": datetime.utcnow(),
                "last_rotated": datetime.utcnow()
            }
            self.secrets.insert_one(secret_record)
            
            logger.info(f"Stored secret {secret_name} for user {user_id}")
            return secret_id
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to store secret: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to store secret: {str(e)}")

    async def retrieve_secret(self, user_id: str, secret_id: str) -> Optional[str]:
        try:
            secret = self.secrets.find_one({"secret_id": secret_id, "user_id": user_id})
            if not secret:
                raise MCPError(code=-32003, message="Secret not found or access denied")
            
            decrypted_value = self.cipher.decrypt(secret["secret_value"].encode()).decode()
            logger.info(f"Retrieved secret {secret['secret_name']} for user {user_id}")
            return decrypted_value
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to retrieve secret: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to retrieve secret: {str(e)}")

    async def rotate_secret(self, secret_id: str, user_id: str) -> None:
        try:
            secret = self.secrets.find_one({"secret_id": secret_id, "user_id": user_id})
            if not secret:
                raise MCPError(code=-32003, message="Secret not found or access denied")
            
            if secret["last_rotated"] + self.rotation_interval > datetime.utcnow():
                raise MCPError(code=-32602, message="Secret rotation not due yet")
            
            new_value = secrets.token_hex(32)
            encrypted_value = self.cipher.encrypt(new_value.encode()).decode()
            self.secrets.update_one(
                {"secret_id": secret_id},
                {"$set": {"secret_value": encrypted_value, "last_rotated": datetime.utcnow()}}
            )
            
            logger.info(f"Rotated secret {secret['secret_name']} for user {user_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to rotate secret: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to rotate secret: {str(e)}")

    async def scan_content(self, content: str) -> List[Dict[str, Any]]:
        try:
            patterns = {
                "github_token": r"ghp_[a-zA-Z0-9]{36}",
                "api_key": r"[a-zA-Z0-9]{32,}",
                "jwt": r"eyJ[a-zA-Z0-9-_=]+\.[a-zA-Z0-9-_=]+\.[a-zA-Z0-9-_.+/=]+"
            }
            detected_secrets = []
            for secret_type, pattern in patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    detected_secrets.append({
                        "type": secret_type,
                        "matches": len(matches),
                        "hash": hashlib.sha256(content.encode()).hexdigest()[:8]
                    })
            if detected_secrets:
                logger.warning(f"Detected {len(detected_secrets)} secrets in content")
            return detected_secrets
        except Exception as e:
            logger.error(f"Failed to scan content: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to scan content: {str(e)}")

    def close(self):
        self.client.close()
