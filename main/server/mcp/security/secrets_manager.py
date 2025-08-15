# main/server/mcp/security/secrets_manager.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

app = FastAPI(title="Vial MCP Secrets Manager")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class Secret(BaseModel):
    user_id: str
    key: str
    value: str

class SecretResponse(BaseModel):
    user_id: str
    key: str
    timestamp: str

class SecretsManager:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.encryption_key = self._generate_encryption_key()
        self.cipher = Fernet(self.encryption_key)

    def _generate_encryption_key(self) -> bytes:
        with self.metrics.track_span("generate_encryption_key"):
            try:
                salt = os.getenv("ENCRYPTION_SALT", "vial_mcp_salt").encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                return base64.urlsafe_b64encode(kdf.derive(os.getenv("JWT_SECRET", "secret_key").encode()))
            except Exception as e:
                handle_generic_error(e, context="generate_encryption_key")
                raise

    def encrypt_secret(self, value: str) -> str:
        with self.metrics.track_span("encrypt_secret"):
            try:
                return self.cipher.encrypt(value.encode()).decode()
            except Exception as e:
                handle_generic_error(e, context="encrypt_secret")
                raise

    def decrypt_secret(self, encrypted_value: str) -> str:
        with self.metrics.track_span("decrypt_secret"):
            try:
                return self.cipher.decrypt(encrypted_value.encode()).decode()
            except Exception as e:
                handle_generic_error(e, context="decrypt_secret")
                raise

secrets_manager = SecretsManager()

@app.post("/secrets", response_model=SecretResponse)
async def store_secret(secret: Secret, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("store_secret", {"user_id": secret.user_id, "key": secret.key}):
        try:
            metrics.verify_token(token)
            encrypted_value = secrets_manager.encrypt_secret(secret.value)
            secret_data = {
                "user_id": secret.user_id,
                "key": secret.key,
                "value": encrypted_value,
                "timestamp": datetime.utcnow()
            }
            db_manager.insert_one("secrets", secret_data)
            return SecretResponse(user_id=secret.user_id, key=secret.key, timestamp=str(secret_data["timestamp"]))
        except Exception as e:
            handle_generic_error(e, context="store_secret")
            raise HTTPException(status_code=500, detail=f"Failed to store secret: {str(e)}")

@app.get("/secrets/{user_id}/{key}", response_model=SecretResponse)
async def retrieve_secret(user_id: str, key: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("retrieve_secret", {"user_id": user_id, "key": key}):
        try:
            metrics.verify_token(token)
            secret = db_manager.find_one("secrets", {"user_id": user_id, "key": key})
            if not secret:
                raise HTTPException(status_code=404, detail="Secret not found")
            return SecretResponse(user_id=secret["user_id"], key=secret["key"], timestamp=str(secret["timestamp"]))
        except Exception as e:
            handle_generic_error(e, context="retrieve_secret")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve secret: {str(e)}")
