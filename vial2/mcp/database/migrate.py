import asyncpg
import os
from .neon_connection import neon_db
from ...mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def migrate():
    try:
        await neon_db.connect()
        await neon_db.execute("""
            CREATE TABLE IF NOT EXISTS vial_logs (
                log_id SERIAL PRIMARY KEY,
                vial_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB,
                node_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Database migration completed")
    except Exception as e:
        error_logger.log_error("database_migration", str(e), str(e.__traceback__), sql_statement="CREATE TABLE vial_logs", sql_error_code=None, params={})
        logger.error(f"Database migration failed: {str(e)}")
        raise
    finally:
        await neon_db.disconnect()

if __name__ == "__main__":
    asyncio.run(migrate())

# xAI Artifact Tags: #vial2 #mcp #database #migration #neon #neon_mcp
