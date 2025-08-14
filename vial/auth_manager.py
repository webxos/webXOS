import jwt
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.secret_key = "vial-mcp-secret-2025"
        self.api_keys = {
            "api-bd9d62ec-a074-4548-8c83-fb054715a870": {
                "user_id": "vial_user",
                "created_at": datetime.now()
            }
        }

    def verify_api_key(self, api_key: str) -> bool:
        try:
            if api_key not in self.api_keys:
                logger.warning(f"Invalid API key attempted: {api_key}")
                return False
            return True
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
            logger.info(f"Generated token for user_id: {self.api_keys[api_key]['user_id']}")
            return token
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            logger.info(f"Token verified for user_id: {payload['user_id']}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Token verification failed: {str(e)}")
