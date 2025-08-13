import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET", "secret-key")
ALGORITHM = "HS256"

logging.basicConfig(filename="db/errorlog.md", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_jwt_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    try:
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        logging.info(f"JWT created for user {user_id}")
        return token
    except Exception as e:
        logging.error(f"Failed to create JWT for {user_id}: {str(e)}")
        raise

def verify_jwt_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logging.info(f"JWT verified for user {payload['sub']}")
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        logging.error("JWT expired")
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        logging.error("Invalid JWT")
        raise ValueError("Invalid token")
