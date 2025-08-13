import os
from dotenv import load_dotenv

load_dotenv()

# Backend configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
VIAL_VERSION = os.getenv('VIAL_VERSION', '2.8')

# Database configuration (for future integration)
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://user:password@localhost:5432/vial_mcp')
REDIS_URI = os.getenv('REDIS_URI', 'redis://localhost:6379')
