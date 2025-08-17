import os
from typing import Optional
from pydantic import BaseModel
import dotenv
import logging

logger = logging.getLogger("mcp.config")
logger.setLevel(logging.INFO)

class DatabaseConfig:
    def __init__(self):
        dotenv.load_dotenv()
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            logger.error("DATABASE_URL not set in environment")
            raise ValueError("DATABASE_URL is required")

class APIConfig(BaseModel):
    github_client_id: str
    github_client_secret: str
    oauth_redirect_uri: str

    def __init__(self):
        dotenv.load_dotenv()
        super().__init__(
            github_client_id=os.getenv("GITHUB_CLIENT_ID", ""),
            github_client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
            oauth_redirect_uri=os.getenv("OAUTH_REDIRECT_URI", "https://webxos.netlify.app/auth/callback")
        )
        if not self.github_client_id or not self.github_client_secret:
            logger.error("GitHub OAuth credentials not set in environment")
            raise ValueError("GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET are required")

class SMTPConfig(BaseModel):
    smtp_server: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    alert_email: str

    def __init__(self):
        dotenv.load_dotenv()
        super().__init__(
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", 587)),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            alert_email=os.getenv("ALERT_EMAIL", "security@webxos.netlify.app")
        )
        if not self.smtp_user or not self.smtp_password:
            logger.error("SMTP credentials not set in environment")
            raise ValueError("SMTP_USER and SMTP_PASSWORD are required")
