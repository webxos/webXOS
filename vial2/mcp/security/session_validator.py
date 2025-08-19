from ..security.session_handler import session_handler
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class SessionValidator:
    async def validate_session(self, session_id: str):
        try:
            session = session_handler.sessions.get(session_id)
            if not session or time.time() > session["expires"]:
                raise ValueError("Session expired or invalid")
            logger.info(f"Validated session {session_id}")
            return True
        except Exception as e:
            error_logger.log_error("session_validate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={session_id})
            logger.error(f"Session validation failed: {str(e)}")
            raise

session_validator = SessionValidator()

# xAI Artifact Tags: #vial2 #mcp #security #session #validator #neon_mcp
