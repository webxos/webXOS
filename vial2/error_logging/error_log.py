import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorLogger:
    def __init__(self, db_path="error_log.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    module TEXT,
                    error_message TEXT,
                    stack_trace TEXT
                )
            """)
            conn.commit()

    def log_error(self, module: str, error_message: str, stack_trace: str = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO errors (timestamp, module, error_message, stack_trace) VALUES (?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(), module, error_message, stack_trace)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log error to SQLite: {str(e)}")

    def get_logs(self, limit: int = 100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT ?", (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {str(e)}")
            return []

error_logger = ErrorLogger()

# xAI Artifact Tags: #vial2 #error_logging #sqlite
