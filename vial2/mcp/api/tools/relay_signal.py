import logging
import uuid
import json
from config.config import DatabaseConfig
from lib.security import SecurityHandler

logger = logging.getLogger(__name__)

class RelaySignal:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.project_id = db.project_id

    async def send_signal(self, user_id: str, project_id: str) -> dict:
        try:
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [relay_signal.py:15] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            await self.db.query(
                "INSERT INTO user_sessions (session_id, user_id, project_id, status) VALUES ($1, $2, $3, $4)",
                [str(uuid.uuid4()), user_id, project_id, "active"]
            )
            await self.security.log_action(user_id, "relay_signal", {"status": "active"})
            logger.info(f"Relay signal sent for user: {user_id} [relay_signal.py:20] [ID:relay_signal_success]")
            return {"status": "success", "user_id": user_id}
        except Exception as e:
            error_message = f"Relay signal failed: {str(e)} [relay_signal.py:25] [ID:relay_signal_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "relay_signal", error_message)
            return {"error": error_message}
