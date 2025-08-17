import os
from dotenv import load_dotenv

load_dotenv()

MCP_CONFIG = {
    "JWT_SECRET": os.getenv("JWT_SECRET", "secret_key_123"),
    "BASE_URL": os.getenv("BASE_URL", "http://localhost:8000")
}
