import os
from dotenv import load_dotenv
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    STACK_AUTH_CLIENT_ID = os.getenv("STACK_AUTH_CLIENT_ID")
    STACK_AUTH_CLIENT_SECRET = os.getenv("STACK_AUTH_CLIENT_SECRET")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    NETLIFY_SITE_ID = os.getenv("NETLIFY_SITE_ID")
    NETLIFY_AUTH_TOKEN = os.getenv("NETLIFY_AUTH_TOKEN")

    @classmethod
    def validate(cls):
        try:
            required = ["DATABASE_URL", "STACK_AUTH_CLIENT_ID", "STACK_AUTH_CLIENT_SECRET", "JWT_SECRET_KEY"]
            for var in required:
                if not getattr(cls, var):
                    raise ValueError(f"Missing environment variable: {var}")
            return True
        except Exception as e:
            error_logger.log_error("config_validation", str(e), str(e.__traceback__))
            logger.error(f"Config validation failed: {str(e)}")
            raise

config = Config()

# xAI Artifact Tags: #vial2 #config #neon_mcp
