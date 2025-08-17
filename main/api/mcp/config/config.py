import os
from pydantic_settings import BaseSettings
from typing import Optional

class DatabaseConfig(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/neondb")
    max_connections: int = 20
    min_connections: int = 1
    connect_timeout: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class APIConfig(BaseSettings):
    max_retries: int = 3
    rate_limit: int = 100
    rate_limit_window: int = 60
    api_key: Optional[str] = os.getenv("API_KEY")
    api_secret: Optional[str] = os.getenv("API_SECRET")
    github_client_id: Optional[str] = os.environ.get("GITHUB_CLIENT_ID")
    github_client_secret: Optional[str] = os.environ.get("GITHUB_CLIENT_SECRET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

database_config = DatabaseConfig()
api_config = APIConfig()
