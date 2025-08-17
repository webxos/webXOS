import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException
from main.api.utils.logging import logger
import os

class AuthManager:
    def __init__(self):
        self.JWT_SECRET = os.getenv("JWT_SECRET", "secret_key_123_change_in_production")

    def generate_token(self, user_id: str):
        """Generate a JWT for a user."""
        try:
            payload = {
                "sub": user_id,
                "exp": (datetime.utcnow() + timedelta(hours=24)).timestamp(),
                "iat": datetime.utcnow().timestamp()
            }
            token = jwt.encode(payload, self.JWT_SECRET, algorithm="HS256")
            logger.info(f"Token generated for user: {user_id}")
            return token
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def verify_token(self, token: str):
        """Verify a JWT."""
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=["HS256"])
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            logger.error("Token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            logger.error("Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")
