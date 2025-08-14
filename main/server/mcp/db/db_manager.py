# server/mcp/db/db_manager.py
import sqlite3
import os
from contextlib import contextmanager
from ..utils.error_handler import handle_db_error

class DatabaseManager:
    def __init__(self, db_path="vial_mcp.db"):
        self.db_path = os.path.join(os.path.dirname(__file__), db_path)
        self._init_db()

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL UNIQUE,
                    status TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id INTEGER,
                    task_type TEXT,
                    task_data TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_id) REFERENCES agents(id)
                )
            """)
            conn.commit()

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.Error as e:
            handle_db_error(e)
            raise
        finally:
            if conn:
                conn.close()

    def save_agent_status(self, agent_name, status):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agents (agent_name, status)
                VALUES (?, ?)
            """, (agent_name, status))
            conn.commit()

    def save_task(self, agent_id, task_type, task_data, status="pending"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks (agent_id, task_type, task_data, status)
                VALUES (?, ?, ?, ?)
            """, (agent_id, task_type, task_data, status))
            conn.commit()
            return cursor.lastrowid

    def get_agent_status(self, agent_name):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM agents WHERE agent_name = ?", (agent_name,))
            result = cursor.fetchone()
            return result[0] if result else None
