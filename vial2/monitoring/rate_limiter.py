from fastapi import HTTPException
import time
import sqlite3
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.db_path = "error_log.db"

    async def check_rate_limit(self, user_id: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN")
                # Clean old requests
                conn.execute("DELETE FROM rate_limits WHERE timestamp < datetime('now', ?)", (f"-{self.window_seconds} seconds",))
                # Count requests in window
                count = conn.execute("SELECT COUNT(*) FROM rate_limits WHERE user_id=?", (user_id,)).fetchone()[0]
                if count >= self.max_requests:
                    conn.execute("ROLLBACK")
                    raise HTTPException(status_code=429, detail={
                        "jsonrpc": "2.0", "error": {"code": -32603, "message": "Rate limit exceeded"}
                    })
                # Log new request
                conn.execute("INSERT INTO rate_limits (user_id, timestamp) VALUES (?, datetime('now'))", (user_id,))
                conn.execute("COMMIT")
                return True
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            error_logger.log_error("rate_limiter_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO rate_limits", sql_error_code=e.sqlite_errorcode, params={"user_id": user_id})
            logger.error(f"Rate limiter DB error: {str(e)}")
            raise HTTPException(status_code=429, detail={
                "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": {"sql_error_code": e.sqlite_errorcode}}
            })
        except Exception as e:
            error_logger.log_error("rate_limiter", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"user_id": user_id})
            logger.error(f"Rate limiter failed: {str(e)}")
            raise HTTPException(status_code=429, detail={
                "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}
            })

rate_limiter = RateLimiter()

# xAI Artifact Tags: #vial2 #monitoring #rate_limiter #sqlite #neon_mcp
