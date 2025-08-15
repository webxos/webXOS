# main/server/mcp/auth/auth_sync.py
from typing import Dict, Any, Optional
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..security.secrets_manager import SecretsManager
from datetime import datetime, timedelta
import jwt
import os
import logging

logger = logging.getLogger("mcp")

class AuthSync:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.sessions = self.db["sessions"]
        self.mfa_tokens = self.db["mfa_tokens"]
        self.secrets_manager = SecretsManager()
        self.secret_key = os.getenv("SECRET_KEY", "your_random_secret_key_32_bytes")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.session_expiry = int(os.getenv("JWT_EXPIRY_MINUTES", 1440)) * 60  # Convert to seconds

    async def create_session(self, user_id: str, mfa_verified: bool = False) -> Dict[str, Any]:
        try:
            if not user_id:
                raise MCPError(code=-32602, message="User ID is required")
            
            session_id = secrets.token_hex(16)
            expiry = datetime.utcnow() + timedelta(seconds=self.session_expiry)
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "mfa_verified": mfa_verified,
                "created_at": datetime.utcnow(),
                "expires_at": expiry
            }
            self.sessions.insert_one(session)
            
            # Generate JWT
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "exp": expiry
            }
            access_token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
            
            logger.info(f"Created session for user {user_id}")
            return {
                "session_id": session_id,
                "access_token": access_token,
                "expires_at": expiry.isoformat()
            }
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to create session: {str(e)}")

    async def initiate_mfa(self, user_id: str, mfa_method: str = "email") -> Dict[str, Any]:
        try:
            if mfa_method not in ["email", "sms", "totp"]:
                raise MCPError(code=-32602, message="Unsupported MFA method")
            
            mfa_token = secrets.token_hex(16)
            mfa_expiry = datetime.utcnow() + timedelta(minutes=5)
            mfa_record = {
                "mfa_token": mfa_token,
                "user_id": user_id,
                "method": mfa_method,
                "created_at": datetime.utcnow(),
                "expires_at": mfa_expiry
            }
            self.mfa_tokens.insert_one(mfa_record)
            
            # Store MFA token securely
            await self.secrets_manager.store_secret(user_id, f"mfa_{mfa_token}", mfa_token)
            
            logger.info(f"Initiated MFA for user {user_id} via {mfa_method}")
            return {
                "mfa_token": mfa_token,
                "method": mfa_method,
                "expires_at": mfa_expiry.isoformat()
            }
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"MFA initiation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to initiate MFA: {str(e)}")

    async def verify_mfa(self, user_id: str, mfa_token: str, mfa_code: str) -> Dict[str, Any]:
        try:
            mfa_record = self.mfa_tokens.find_one({"mfa_token": mfa_token, "user_id": user_id})
            if not mfa_record or mfa_record["expires_at"] < datetime.utcnow():
                raise MCPError(code=-32003, message="Invalid or expired MFA token")
            
            # Verify MFA code (mocked for simplicity; integrate with actual MFA service)
            stored_mfa = await self.secrets_manager.retrieve_secret(user_id, f"mfa_{mfa_token}")
            if stored_mfa != mfa_token:
                raise MCPError(code=-32003, message="Invalid MFA code")
            
            # Update session to mark MFA as verified
            session = self.sessions.find_one({"user_id": user_id})
            if not session:
                raise MCPError(code=-32003, message="No active session found")
            
            self.sessions.update_one(
                {"session_id": session["session_id"]},
                {"$set": {"mfa_verified": True}}
            )
            
            logger.info(f"MFA verified for user {user_id}")
            return {"status": "success", "session_id": session["session_id"]}
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"MFA verification failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to verify MFA: {str(e)}")

    async def revoke_session(self, session_id: str, user_id: str) -> None:
        try:
            result = self.sessions.delete_one({"session_id": session_id, "user_id": user_id})
            if result.deleted_count == 0:
                raise MCPError(code=-32003, message="Session not found or access denied")
            logger.info(f"Revoked session {session_id} for user {user_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Session revocation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to revoke session: {str(e)}")

    def close(self):
        self.client.close()
