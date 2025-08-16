import sqlite3
import logging
import os
from datetime import datetime

# Configure file-based logging
logging.basicConfig(
    filename="main/vial_mcp.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def init_db():
    """Initialize SQLite database for error logging."""
    conn = sqlite3.connect("main/errors.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            endpoint TEXT,
            user_id TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_info(message: str, endpoint: str = None, user_id: str = None):
    """Log info message to file and SQLite."""
    logging.info(message)
    conn = sqlite3.connect("main/errors.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO error_logs (timestamp, level, message, endpoint, user_id) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), "INFO", message, endpoint, user_id)
    )
    conn.commit()
    conn.close()

def log_error(message: str, endpoint: str = None, user_id: str = None):
    """Log error message to file and SQLite."""
    logging.error(message)
    conn = sqlite3.connect("main/errors.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO error_logs (timestamp, level, message, endpoint, user_id) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), "ERROR", message, endpoint, user_id)
    )
    conn.commit()
    conn.close()

# Initialize database on module import
if not os.path.exists("main/errors.db"):
    init_db()
