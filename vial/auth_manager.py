import jwt
import uuid
from typing import Tuple, Dict
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.secret_key = os.getenv("API_TOKEN", "secret-token")

    def authenticate(self, network_id: str, session_id: str) -> Tuple[str, str]:
        """Authenticate a session and return JWT token and address."""
        try:
            token = jwt.encode(
                {"network_id": network_id, "session_id": session_id, "uuid": str(uuid.uuid4())},
                self.secret_key,
                algorithm="HS256"
            )
            address = f"0x{uuid.uuid4().hex[:40]}"
            self.sessions[token] = {"network_id": network_id, "session_id": session_id}
            logger.info(f"Authenticated: {token} for network {network_id}")
            return token, address
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T05:46:00Z]** Authentication error: {str(e)}\n")
            raise

    def validate_token(self, token: str) -> bool:
        """Validate a JWT token."""
        try:
            jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return token in self.sessions
        except jwt.InvalidTokenError:
            return False

    def validate_session(self, token: str, network_id: str) -> bool:
        """Validate session for a network ID."""
        session = self.sessions.get(token)
        return session and session["network_id"] == network_id

    def void_session(self, token: str):
        """Void a session."""
        if token in self.sessions:
            del self.sessions[token]
            logger.info(f"Voided session: {token}")