import os
from main.api.utils.logging import logger

class MCPConfig:
    def __init__(self):
        self.MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
        self.JWT_SECRET = os.getenv("JWT_SECRET", "secret_key_123_change_in_production")
        self.VIAL_IDS = [f"vial{i+1}" for i in range(4)]
        self.POW_REWARD = 10.0
        self.MAX_VIALS = 4
        self.TRAINING_TIMEOUT = 300  # seconds

    def validate(self):
        """Validate configuration settings."""
        try:
            if not self.MONGO_URL.startswith("mongodb://"):
                raise ValueError("Invalid MONGO_URL format")
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required")
            if not self.JWT_SECRET:
                raise ValueError("JWT_SECRET is required")
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

config = MCPConfig()
