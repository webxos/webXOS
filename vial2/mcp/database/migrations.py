from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import asyncpg

logger = logging.getLogger(__name__)

async def migrate_database():
    try:
        await neon_db.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                traceback TEXT,
                sql_statement TEXT,
                sql_error_code TEXT,
                params JSONB
            );
            CREATE TABLE IF NOT EXISTS mcp_sessions (
                session_id UUID PRIMARY KEY,
                vial_id TEXT,
                user_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_error_logs_timestamp ON error_logs(timestamp);
        """)
        logger.info("Database migration completed")
    except Exception as e:
        error_logger.log_error("database_migrate", str(e), str(e.__traceback__), sql_statement=str(e.__cause__), sql_error_code=None, params={})
        logger.error(f"Database migration failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #database #migrations #neon_mcp
