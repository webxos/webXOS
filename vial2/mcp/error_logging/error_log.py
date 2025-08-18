import sqlite3
import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

def log_error(error_type: str, error_message: str, traceback_str: str, sql_statement: Optional[str] = None, sql_error_code: Optional[int] = None, params: dict = None) -> None:
    try:
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    traceback TEXT NOT NULL,
                    sql_statement TEXT,
                    sql_error_code INTEGER,
                    params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO error_logs (error_type, error_message, traceback, sql_statement, sql_error_code, params)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (error_type, error_message, traceback_str, sql_statement, sql_error_code, json.dumps(params) if params else None))
    except sqlite3.Error as e:
        logger.error(f"Failed to log error to SQLite: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in error logging: {str(e)}")

# xAI Artifact Tags: #vial2 #mcp #error #logging #sqlite #neon_mcp
