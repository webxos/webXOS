import os
from typing import Optional
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

class DatabaseConfig:
    def __init__(self):
        self.url = os.getenv("NEON_DATABASE_URL")

    async def query(self, query: str, params: list = None):
        # Placeholder for actual database connection logic
        pass

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    ssl_cert_path: Optional[str] = os.getenv("SSL_CERT_PATH")
    ssl_key_path: Optional[str] = os.getenv("SSL_KEY_PATH")
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID")
    jwt_secret: str = os.getenv("JWT_SECRET")

limiter = Limiter(key_func=get_remote_address)
batch_sync_limiter = limiter.limit("5/minute")  # Limit batchSync to 5 requests per minute per IP
