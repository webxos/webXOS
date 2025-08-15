# main/server/mcp/auth/auth_manager.py
from typing import Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel
from webauthn import generate_registration_options, verify_registration_response
from webauthn.helpers.structs import RegistrationCredential
from ..utils.mcp_error_handler import MCPError, handle_mcp_error
from ..wallet.webxos_wallet import WalletService
import os
import base64
import json

class AuthCredentials(BaseModel):
    username: str
    password: Optional[str] = None
    webauthn_credential: Optional[RegistrationCredential] = None
    wallet_address: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str

class AuthManager:
    def __init__(self):
        self.wallet_service = WalletService()
        self.rp_id = os.getenv("RP_ID", "vial-mcp.local")
        self.rp_name = "Vial MCP Controller"
        self.challenge_timeout = 60000  # 60 seconds

    async def authenticate_user(self, credentials: AuthCredentials) -> AuthResponse:
        try:
            if credentials.webauthn_credential:
                # WebAuthn authentication
                registration_options = generate_registration_options(
                    rp_id=self.rp_id,
                    rp_name=self.rp_name,
                    user_id=credentials.username,
                    user_name=credentials.username,
                    attestation="direct"
                )
                verification = verify_registration_response(
                    credential=credentials.webauthn_credential,
                    expected_challenge=base64.b64encode(registration_options.challenge),
                    expected_rp_id=self.rp_id,
                    expected_origin=f"https://{self.rp_id}"
                )
                user_id = credentials.username
                access_token = self._generate_token(user_id)
            elif credentials.wallet_address:
                # Wallet-based authentication
                wallet_verified = await self.wallet_service.verify_wallet(credentials.wallet_address)
                if not wallet_verified:
                    raise MCPError(code=-32001, message="Invalid wallet address")
                user_id = credentials.wallet_address
                access_token = self._generate_token(user_id)
            elif credentials.username and credentials.password:
                # OAuth 3.0 password grant
                if not self._verify_password(credentials.username, credentials.password):
                    raise MCPError(code=-32001, message="Invalid username or password")
                user_id = credentials.username
                access_token = self._generate_token(user_id)
            else:
                raise MCPError(code=-32602, message="Invalid authentication credentials")
            
            return AuthResponse(
                access_token=access_token,
                token_type="Bearer",
                user_id=user_id
            )
        except MCPError as e:
            raise HTTPException(status_code=401, detail=handle_mcp_error(e))
        except Exception as e:
            raise MCPError(code=-32603, message=f"Authentication error: {str(e)}")

    def _generate_token(self, user_id: str) -> str:
        # Simplified token generation (in production, use JWT or similar)
        return base64.b64encode(f"{user_id}:{os.urandom(16).hex()}".encode()).decode()

    def _verify_password(self, username: str, password: str) -> bool:
        # Placeholder for password verification (use secure storage in production)
        return password == "secure_password"  # Replace with proper password hashing/check

    async def validate_token(self, token: str) -> Dict[str, str]:
        try:
            decoded = base64.b64decode(token).decode().split(":")
            if len(decoded) != 2:
                raise MCPError(code=-32001, message="Invalid token format")
            return {"user_id": decoded[0]}
        except Exception as e:
            raise MCPError(code=-32001, message=f"Token validation failed: {str(e)}")
