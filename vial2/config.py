import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_EzPpBWkGdm69@ep-sparkling-thunder-aetjtveu-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require")
    STACK_AUTH_CLIENT_ID = os.getenv("STACK_AUTH_CLIENT_ID", "142ad169-5d57-4be3-bf41-6f3cd0a9ae1d")
    STACK_AUTH_SECRET_KEY = os.getenv("STACK_AUTH_SECRET_KEY", "ssk_jg4mmhab0d0ga2krj1sskadmnkaagcxy7nxwbaeagkbjg")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_key")
    ALLOWED_ORIGINS = [
        "https://webxos.netlify.app",
        "http://localhost:3000",
        "http://localhost:8000"
    ]

config = Config()

# xAI Artifact Tags: #vial2 #config #neon_mcp
