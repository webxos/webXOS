from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    MONGO_URI: str = "mongodb://localhost:27017/webxos"
    JWT_SECRET: str = "your_jwt_secret_here"  # Replace with secure key
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
