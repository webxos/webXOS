from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class SecretsManager:
    def __init__(self):
        self.secrets = {}

    def load_secret(self, key: str):
        try:
            value = os.getenv(key)
            if not value:
                raise ValueError(f"Secret {key} not found in environment")
            self.secrets[key] = value
            return value
        except Exception as e:
            error_logger.log_error("secrets", f"Failed to load secret {key}: {str(e)}", str(e.__traceback__))
            logger.error(f"Failed to load secret: {str(e)}")
            raise

    def get_secret(self, key: str):
        try:
            return self.secrets.get(key, self.load_secret(key))
        except Exception as e:
            error_logger.log_error("secrets", f"Failed to get secret {key}: {str(e)}", str(e.__traceback__))
            logger.error(f"Failed to get secret: {str(e)}")
            raise

secrets_manager = SecretsManager()

# xAI Artifact Tags: #vial2 #config #secrets #neon_mcp
