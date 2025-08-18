from fastapi import APIRouter, Depends
import sqlite3
import time
from ..error_logging.error_log import error_logger
import logging

router = APIRouter(prefix="/mcp/api", tags=["performance"])

logger = logging.getLogger(__name__)

@router.get("/performance")
async def analyze_performance():
    try:
        start_time = time.time()
        with sqlite3.connect("error_log.db") as conn:
            # Simulate high concurrency
            for _ in range(10):
                conn.execute("INSERT INTO errors (module, message) VALUES (?, ?)", ("perf_test", "test"))
            # Analyze query plan
            plan = conn.execute("EXPLAIN QUERY PLAN SELECT * FROM errors").fetchall()
        latency = time.time() - start_time
        return {"status": "success", "latency": latency, "query_plan": plan}
    except sqlite3.OperationalError as e:
        error_logger.log_error("performance_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO errors", sql_error_code=e.sqlite_errorcode, params=None)
        logger.error(f"Performance test failed due to DB lock: {str(e)}")
        raise HTTPException(status_code=429, detail={"error": {"code": -32603, "message": "Database lock", "data": str(e)}})
    except Exception as e:
        error_logger.log_error("performance", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Performance analysis failed: {str(e)}")
        raise HTTPException(status_code=400, detail={"error": {"code": -32603, "message": str(e)}})

# xAI Artifact Tags: #vial2 #monitoring #performance #sqlite #neon_mcp
