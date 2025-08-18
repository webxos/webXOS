import sqlite3
import traceback
import logging

logger = logging.getLogger(__name__)

class ErrorLogger:
    def __init__(self, db_path="error_log.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        module TEXT,
                        message TEXT,
                        traceback TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error log database initialization failed: {str(e)}")

    def log_error(self, module: str, message: str, traceback_str: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO errors (module, message, traceback) VALUES (?, ?, ?)",
                    (module, message, traceback_str)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging failed: {str(e)}")

error_logger = ErrorLogger()

# xAI Artifact Tags: #vial2 #error_logging #neon_mcp
