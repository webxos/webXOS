from ..database import get_db
from ..error_logging.error_log import error_logger
import logging
import asyncpg

logger = logging.getLogger(__name__)

async def run_migrations():
    try:
        db = await get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                wallet_address VARCHAR(42) UNIQUE NOT NULL,
                api_key VARCHAR(256),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS wallets (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                address VARCHAR(42) UNIQUE NOT NULL,
                balance FLOAT DEFAULT 0.0,
                hash VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            ALTER TABLE users ENABLE ROW LEVEL SECURITY;
            CREATE POLICY user_access ON users FOR ALL TO PUBLIC USING (true);
            ALTER TABLE wallets ENABLE ROW LEVEL SECURITY;
            CREATE POLICY wallet_access ON wallets FOR ALL TO PUBLIC USING (true);
        """)
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("migrations", str(e), str(e.__traceback__))
        logger.error(f"Migration failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #migrations #neon_mcp
