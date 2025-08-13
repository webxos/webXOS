import jwt
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.secret_key = "vial-mcp-secret-2025"
        self.api_keys = {"api-b2116602-3486-42a5-801b-a176bff044b7": {"user_id": "vial_user", "created_at": datetime.now()}}
        
    def verify_api_key(self, api_key: str) -> bool:
        try:
            return api_key in self.api_keys
        except Exception as e:
            logger.error(f"API key verification failed: {str(e)}")
            return False

    def generate_token(self, api_key: str) -> str:
        try:
            if not self.verify_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            payload = {
                "user_id": self.api_keys[api_key]["user_id"],
                "exp": datetime.utcnow() + timedelta(hours=24)
            }
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            return token
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Token verification failed: {str(e)}")
