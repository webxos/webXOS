import asyncpg
from ..config import config
from ..error_logging.error_log import error_logger
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_db_pool():
    try:
        pool = await asyncpg.create_pool(config.DATABASE_URL, max_size=10)
        async with pool.acquire() as conn:
            yield conn
    except Exception as e:
        error_logger.log_error("helpers", f"Database pool connection failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Database pool connection failed: {str(e)}")
        raise
    finally:
        await pool.close()

async def log_event(event_type: str, message: str, db):
    try:
        async with db:
            await db.execute(
                "INSERT INTO logs (event_type, message, timestamp) VALUES ($1, $2, $3)",
                event_type, message, datetime.utcnow()
            )
    except Exception as e:
        error_logger.log_error("helpers", f"Event logging failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Event logging failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #utils #neon_mcp
