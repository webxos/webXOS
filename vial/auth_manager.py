import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionError
import jwt
from datetime import datetime, timedelta
import hashlib
import redis
import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(filename='/db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://user:password@localhost:5432/vial_mcp')
REDIS_URI = os.getenv('REDIS_URI', 'redis://localhost:6379')
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
VIAL_VERSION = '2.8'

app = FastAPI(title="Vial MCP Authentication Service", version=VIAL_VERSION)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Database connections
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client['vial_mcp']
    mongo_client.admin.command('ping')
    logger.info("MongoDB connection established")
except ConnectionError as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    raise Exception(f"MongoDB connection failed: {str(e)}")

try:
    pg_conn = psycopg2.connect(POSTGRES_URI)
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(255) PRIMARY KEY,
            wallet_address VARCHAR(64),
            wallet_hash VARCHAR(64),
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    pg_conn.commit()
    logger.info("PostgreSQL connection established and users table created")
except OperationalError as e:
    logger.error(f"PostgreSQL connection failed: {str(e)}")
    raise Exception(f"PostgreSQL connection failed: {str(e)}")

try:
    redis_client = redis.Redis.from_url(REDIS_URI, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except redis.ConnectionError as e:
    logger.error(f"Redis connection failed: {str(e)}")
    raise Exception(f"Redis connection failed: {str(e)}")

class AuthRequest(BaseModel):
    userId: str

class AuthResponse(BaseModel):
    apiKey: str
    walletAddress: str
    walletHash: str

def generate_jwt(user_id: str) -> str:
    try:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        redis_client.setex(f"session:{user_id}", timedelta(hours=24), token)
        logger.info(f"Generated JWT for user {user_id}")
        return token
    except Exception as e:
        logger.error(f"JWT generation failed for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JWT generation failed: {str(e)}")

def verify_jwt(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user_id = payload['user_id']
        stored_token = redis_client.get(f"session:{user_id}")
        if stored_token != token:
            logger.error(f"Invalid session for user {user_id}")
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        logger.info(f"Verified JWT for user {user_id}")
        return user_id
    except jwt.PyJWTError as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@app.get("/auth/health")
async def health_check():
    try:
        mongo_client.admin.command('ping')
        pg_cursor.execute("SELECT 1")
        redis_client.ping()
        return {"status": "healthy", "mongo": True, "postgres": True, "redis": True, "version": VIAL_VERSION}
    except (ConnectionError, OperationalError, redis.ConnectionError) as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "mongo": False, "postgres": False, "redis": False, "version": VIAL_VERSION}

@app.post("/auth/login", response_model=AuthResponse)
async def login(auth: AuthRequest):
    try:
        user_id = auth.userId
        wallet_address = hashlib.sha256(user_id.encode()).hexdigest()
        wallet_hash = hashlib.sha256((user_id + str(datetime.utcnow())).encode()).hexdigest()
        token = generate_jwt(user_id)
        pg_cursor.execute(
            """
            INSERT INTO users (user_id, wallet_address, wallet_hash, last_login)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) UPDATE
            SET wallet_address = EXCLUDED.wallet_address,
                wallet_hash = EXCLUDED.wallet_hash,
                last_login = EXCLUDED.last_login
            """,
            (user_id, wallet_address, wallet_hash, datetime.utcnow())
        )
        pg_conn.commit()
        logger.info(f"User {user_id} logged in successfully")
        return AuthResponse(apiKey=token, walletAddress=wallet_address, walletHash=wallet_hash)
    except Exception as e:
        logger.error(f"Login error for user {auth.userId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/auth/api-key/generate", response_model=AuthResponse)
async def generate_api_key(auth: AuthRequest):
    try:
        user_id = auth.userId
        wallet_address = hashlib.sha256(user_id.encode()).hexdigest()
        wallet_hash = hashlib.sha256((user_id + str(datetime.utcnow())).encode()).hexdigest()
        token = generate_jwt(user_id)
        pg_cursor.execute(
            """
            INSERT INTO users (user_id, wallet_address, wallet_hash, last_login)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) UPDATE
            SET wallet_address = EXCLUDED.wallet_address,
                wallet_hash = EXCLUDED.wallet_hash,
                last_login = EXCLUDED.last_login
            """,
            (user_id, wallet_address, wallet_hash, datetime.utcnow())
        )
        pg_conn.commit()
        logger.info(f"Generated API key for user {user_id}")
        return AuthResponse(apiKey=token, walletAddress=wallet_address, walletHash=wallet_hash)
    except Exception as e:
        logger.error(f"API key generation error for user {auth.userId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API key generation failed: {str(e)}")

@app.get("/auth/validate/{token}")
async def validate_token(token: str):
    user_id = verify_jwt(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_id": user_id}

@app.delete("/auth/logout")
async def logout(user_id: str = Depends(verify_jwt)):
    try:
        redis_client.delete(f"session:{user_id}")
        logger.info(f"User {user_id} logged out successfully")
        return {"status": "logged out"}
    except Exception as e:
        logger.error(f"Logout error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        pg_conn.close()
        redis_client.close()
        mongo_client.close()
        logger.info("Shutdown: Closed database connections")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
