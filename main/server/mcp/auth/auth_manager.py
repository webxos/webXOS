# main/server/mcp/auth/auth_manager.py
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
from pymongo import MongoClient
import os
from fastapi.security import OAuth2PasswordBearer
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from ..db.db_manager import DBManager

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "secret_key")
        self.algorithm = "HS256"
        self.token_expiry = int(os.getenv("TOKEN_EXPIRY_MINUTES", 30))
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.metrics = PerformanceMetrics()
        self.db_manager = DBManager()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

    def create_token(self, user_id: str, additional_data: Dict = None) -> str:
        with self.metrics.track_span("create_token", {"user_id": user_id}):
            try:
                payload = {
                    "sub": user_id,
                    "iat": datetime.utcnow(),
                    "exp": datetime.utcnow() + timedelta(minutes=self.token_expiry)
                }
                if additional_data:
                    payload.update(additional_data)
                token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
                self.db_manager.insert_one("sessions", {
                    "user_id": user_id,
                    "token": token,
                    "created_at": datetime.utcnow(),
                    "expires_at": payload["exp"]
                })
                return token
            except Exception as e:
                handle_generic_error(e, context="create_token")
                raise

    def verify_token(self, token: str) -> Optional[Dict]:
        with self.metrics.track_span("verify_token", {"token": token[:10] + "..."}):
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                session = self.db_manager.find_one("sessions", {"token": token})
                if not session or session["expires_at"] < datetime.utcnow():
                    raise ValueError("Invalid or expired token")
                return payload
            except Exception as e:
                handle_generic_error(e, context="verify_token")
                return None

    def invalidate_token(self, token: str) -> bool:
        with self.metrics.track_span("invalidate_token", {"token": token[:10] + "..."}):
            try:
                result = self.db_manager.delete_one("sessions", {"token": token})
                return result > 0
            except Exception as e:
                handle_generic_error(e, context="invalidate_token")
                return False

    async def webauthn_challenge(self, user_id: str) -> Dict:
        with self.metrics.track_span("webauthn_challenge", {"user_id": user_id}):
            try:
                challenge = os.urandom(32).hex()
                self.db_manager.update_one("users", {"user_id": user_id}, {"webauthn_challenge": challenge})
                return {"challenge": challenge}
            except Exception as e:
                handle_generic_error(e, context="webauthn_challenge")
                raise
