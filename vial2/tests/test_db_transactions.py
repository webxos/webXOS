import pytest
import sqlite3
from ..db_transactions import SQLiteTransactionManager
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_transaction_success():
    try:
        manager = SQLiteTransactionManager()
        queries = [
            ("INSERT INTO errors (module, message) VALUES (?, ?)", ("test", "success")),
            ("INSERT INTO errors (module, message) VALUES (?, ?)", ("test", "success2"))
        ]
        result = manager.execute_transaction(queries)
        assert result["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_transaction_success", str(e), str(e.__traceback__), sql_statement="INSERT INTO errors", sql_error_code=None, params=None)
        logger.error(f"Transaction success test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_transaction_rollback():
    try:
        manager = SQLiteTransactionManager()
        queries = [
            ("INSERT INTO errors (module, message) VALUES (?, ?)", ("test", "success")),
            ("INSERT INTO errors (module, message) VALUES (?, ?)", ("test", None))  # Constraint violation
        ]
        with pytest.raises(sqlite3.Error):
            manager.execute_transaction(queries)
        # Verify rollback
        with sqlite3.connect("error_log.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM errors WHERE message='success'").fetchone()[0]
            assert count == 0
    except Exception as e:
        error_logger.log_error("test_transaction_rollback", str(e), str(e.__traceback__), sql_statement="INSERT INTO errors", sql_error_code=None, params=None)
        logger.error(f"Transaction rollback test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #db_transactions #sqlite #neon_mcp
