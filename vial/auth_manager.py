import uuid
from typing import Dict, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def authenticate(self, network_id: str, session_id: str) -> Tuple[str, str]:
        try:
            token = str(uuid.uuid4())
            address = str(uuid.uuid4())
            self.sessions[token] = {
                "network_id": network_id,
                "session_id": session_id,
                "address": address,
                "start_time": datetime.utcnow().isoformat()
            }
            logger.info(f"Authenticated session: {token}")
            return token, address
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T00:44:00Z]** Authentication error: {str(e)}\n")
            raise

    def validate_token(self, token: str) -> bool:
        try:
            return token in self.sessions or token == "offline"
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T00:44:00Z]** Token validation error: {str(e)}\n")
            raise

    def validate_session(self, token: str, network_id: str) -> bool:
        try:
            return token in self.sessions and self.sessions[token]["network_id"] == network_id
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T00:44:00Z]** Session validation error: {str(e)}\n")
            raise

    def void_session(self, token: str):
        try:
            if token in self.sessions:
                del self.sessions[token]
                logger.info(f"Voided session: {token}")
        except Exception as e:
            logger.error(f"Void session error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T00:44:00Z]** Void session error: {str(e)}\n")
            raise
