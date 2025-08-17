import os
from dotenv import load_dotenv

load_dotenv()

MCP_CONFIG = {
    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "MONGO_URL": os.getenv("MONGO_URL", "mongodb://localhost:27017"),
    "JWT_SECRET": os.getenv("JWT_SECRET", "secret_key_123"),
    "BASE_URL": os.getenv("BASE_URL", "https://webxos-mcp-gateway.onrender.com"),
    "FALLBACK_URL": os.getenv("FALLBACK_URL", "http://localhost:8000")
}
