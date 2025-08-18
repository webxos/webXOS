import sqlite3
import json
import traceback
import logging
import time
from ..config import Config

logger = logging.getLogger(__name__)

class ErrorLogger:
    def __init__(self, db_path="error_log.db"):
        self.db_path = db_path
        self.node_id = Config.NETLIFY_SITE_ID or "unknown_node"
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        module TEXT,
                        message TEXT,
                        traceback TEXT,
                        sql_statement TEXT,
                        sql_error_code INTEGER,
                        params TEXT,
                        node_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("COMMIT")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error log database initialization failed: {str(e)}")
            raise

    def log_error(self, module: str, message: str, traceback_str: str, sql_statement: str = None, sql_error_code: int = None, params: dict = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO errors (module, message, traceback, sql_statement, sql_error_code, params, node_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (module, message, traceback_str, sql_statement, sql_error_code, json.dumps(params) if params else None, self.node_id)
                )
                conn.execute("COMMIT")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            logger.error(f"Error logging failed: {str(e)}")
            raise

error_logger = ErrorLogger()

# xAI Artifact Tags: #vial2 #error_logging #sqlite #neon_mcp
