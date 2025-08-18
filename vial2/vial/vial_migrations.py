import sqlite3
from ..database import get_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def run_migrations():
    try:
        db = await get_db()
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("BEGIN")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vial_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vial_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    node_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("COMMIT")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS vials (
                vial_id TEXT PRIMARY KEY,
                status TEXT,
                config JSONB,
                quantum_state JSONB
            )
        """)
        logger.info("Vial migrations completed")
        return {"status": "success"}
    except sqlite3.Error as e:
        conn.execute("ROLLBACK")
        error_logger.log_error("vial_migrations_db", str(e), str(e.__traceback__), sql_statement="CREATE TABLE vial_logs", sql_error_code=e.sqlite_errorcode, params=None)
        logger.error(f"Vial migrations failed: {str(e)}")
        raise
    except Exception as e:
        error_logger.log_error("vial_migrations", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Vial migrations failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #vial #migrations #sqlite #neon_mcp
