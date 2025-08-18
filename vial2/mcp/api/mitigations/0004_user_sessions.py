from config.config import DatabaseConfig
import logging

logger = logging.getLogger(__name__)

async def apply():
    try:
        db = DatabaseConfig()
        await db.connect()
        await db.query("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id UUID PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            GRANT SELECT, INSERT, UPDATE ON user_sessions TO replication_user;
            ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
            CREATE POLICY session_access ON user_sessions
                USING (project_id = 'twilight-art-21036984' AND user_id = current_user);
        """)
        await db.disconnect()
        logger.info("Migration 0004 applied [0004_user_sessions.py:20] [ID:migration_success]")
    except Exception as e:
        logger.error(f"Migration 0004 failed: {str(e)} [0004_user_sessions.py:25] [ID:migration_error]")
        raise

async def rollback():
    try:
        db = DatabaseConfig()
        await db.connect()
        await db.query("""
            DROP TABLE IF EXISTS user_sessions;
        """)
        await db.disconnect()
        logger.info("Migration 0004 rolled back [0004_user_sessions.py:30] [ID:rollback_success]")
    except Exception as e:
        logger.error(f"Rollback 0004 failed: {str(e)} [0004_user_sessions.py:35] [ID:rollback_error]")
        raise
