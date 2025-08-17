import os
from pydantic_settings import BaseSettings

class MCPConfig(BaseSettings):
    ENVIRONMENT: str = "development"
    API_URL: str = "https://vial-mcp-backend.onrender.com"
    SECRET_KEY: str = "your_jwt_secret_here"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    MONGO_URI: str = "mongodb://localhost:27017/webxos"

    class Config:
        env_file = "main/.env"
        env_file_encoding = "utf-8"

config = MCPConfig()
