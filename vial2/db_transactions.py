import sqlite3
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class SQLiteTransactionManager:
    def __init__(self, db_path="error_log.db"):
        self.db_path = db_path

    def execute_transaction(self, queries: list):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN")
                for query, params in queries:
                    conn.execute(query, params)
                conn.execute("COMMIT")
                return {"status": "success"}
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            error_logger.log_error("db_transaction", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=e.sqlite_errorcode, params=params)
            logger.error(f"Transaction failed: {str(e)}")
            raise

# xAI Artifact Tags: #vial2 #db_transactions #sqlite #neon_mcp
