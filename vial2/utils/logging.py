import logging
from logging.handlers import RotatingFileHandler
from ..error_logging.error_log import error_logger
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_advanced_logging():
    try:
        handler = RotatingFileHandler("vial2.log", maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("logging", f"Advanced logging setup failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Advanced logging setup failed: {str(e)}")
        raise

async def log_system_event(event: str, details: str, db):
    try:
        logger.info(f"System event: {event} - {details}")
        async with db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                event, details, datetime.utcnow()
            )
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("logging", f"System event logging failed: {str(e)}", str(e.__traceback__))
        logger.error(f"System event logging failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #utils #logging #neon_mcp
