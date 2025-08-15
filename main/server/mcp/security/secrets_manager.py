# main/server/mcp/security/security_manager.py
import asyncio
from typing import Dict, Optional
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import kyber
import re
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
import os

class PromptGuard:
    def __init__(self):
        self.injection_patterns = [
            r"(?i)ignore.*instructions",
            r"(?i)system:.*",
            r"(?i)prompt:.*",
            r"(?i)execute.*code",
            r"\|.*\|",
            r"<\|.*\|>",
        ]
        self.max_length = 1000

    async def validate_async(self, text: str) -> Dict:
        if len(text) > self.max_length:
            return {"threat_detected": True, "details": "Input too long"}
        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                return {"threat_detected": True, "details": f"Pattern matched: {pattern}"}
        return {"threat_detected": False, "details": "Input clean"}

class SecurityManager:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.prompt_guard = PromptGuard()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
        self.kyber_keypair = kyber.Kyber512().generate_keypair()

    async def validate_input(self, text: str, user_id: str) -> bool:
        with self.metrics.track_span("validate_input", {"user_id": user_id}):
            try:
                result = await self.prompt_guard.validate_async(text)
                if result["threat_detected"]:
                    self.metrics.record_error("prompt_injection", result["details"])
                    raise HTTPException(400, detail="Potentially malicious input detected")
                return True
            except Exception as e:
                handle_generic_error(e, context="validate_input")
                raise

    async def encrypt_sensitive_data(self, data: str) -> bytes:
        with self.metrics.track_span("encrypt_data", {}):
            try:
                ciphertext = kyber.Kyber512().encrypt(data.encode(), self.kyber_keypair.public_key)
                return ciphertext
            except Exception as e:
                handle_generic_error(e, context="encrypt_data")
                raise

    async def decrypt_sensitive_data(self, ciphertext: bytes) -> str:
        with self.metrics.track_span("decrypt_data", {}):
            try:
                plaintext = kyber.Kyber512().decrypt(ciphertext, self.kyber_keypair.private_key)
                return plaintext.decode()
            except Exception as e:
                handle_generic_error(e, context="decrypt_data")
                raise

    async def verify_oauth3_token(self, token: str = Depends(oauth2_scheme)) -> Dict:
        with self.metrics.track_span("verify_oauth3_token", {}):
            try:
                # Placeholder for OAuth 3.0 with PKCE verification
                # In a full implementation, verify PKCE code_challenge and code_verifier
                payload = self.metrics.verify_token(token)  # Reuse existing JWT verification
                if not payload.get("code_challenge"):
                    raise HTTPException(401, detail="Missing PKCE code challenge")
                return payload
            except Exception as e:
                handle_generic_error(e, context="verify_oauth3_token")
                raise HTTPException(401, detail=f"Invalid token: {str(e)}")
