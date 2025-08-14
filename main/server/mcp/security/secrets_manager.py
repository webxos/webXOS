import logging
import os
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class SecretRequest(BaseModel):
    key: str
    value: str

class SecretsManager:
    """Manages secrets for Vial MCP."""
    def __init__(self, error_handler: ErrorHandler = None):
        """Initialize SecretsManager with environment-based secrets.

        Args:
            error_handler (ErrorHandler): Error handler instance.
        """
        self.secrets = {}
        self.error_handler = error_handler or ErrorHandler()
        self.load_secrets()
        logger.info("SecretsManager initialized")

    def load_secrets(self):
        """Load secrets from environment variables."""
        try:
            self.secrets["JWT_SECRET"] = os.getenv("JWT_SECRET", "default_jwt_secret")
            self.secrets["GROK_API_KEY"] = os.getenv("GROK_API_KEY", "default_grok_key")
            logger.info("Secrets loaded from environment")
        except Exception as e:
            logger.error(f"Failed to load secrets: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [SecretsManager] Failed to load secrets: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Failed to load secrets: {str(e)}")

    async def store_secret(self, request: SecretRequest) -> dict:
        """Store a secret.

        Args:
            request (SecretRequest): Secret key-value pair.

        Returns:
            dict: Store result.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            self.secrets[request.key] = request.value
            logger.info(f"Stored secret: {request.key}")
            return {"status": "success", "key": request.key}
        except Exception as e:
            self.error_handler.handle_exception("/api/secrets/store", request.key, e)

    async def retrieve_secret(self, key: str) -> dict:
        """Retrieve a secret.

        Args:
            key (str): Secret key.

        Returns:
            dict: Secret value.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            value = self.secrets.get(key)
            if not value:
                error_msg = f"Secret not found: {key}"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/secrets/retrieve", key, Exception(error_msg))
            logger.info(f"Retrieved secret: {key}")
            return {"key": key, "value": value}
        except Exception as e:
            self.error_handler.handle_exception("/api/secrets/retrieve", key, e)
