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
            conn.execute("PRAGMA busy_timeout = 5000")  # Handle locks
            # Simulate high concurrency
            for _ in range(10):
                conn.execute("INSERT INTO errors (module, message, node_id) VALUES (?, ?, ?)", ("perf_test", "test", "node1"))
            # Analyze query plan
            plan = conn.execute("EXPLAIN QUERY PLAN SELECT * FROM errors WHERE node_id='node1'").fetchall()
        latency = time.time() - start_time
        # Log performance metrics
        error_logger.log_error("performance_metrics", "Performance analysis completed", "", sql_statement=None, sql_error_code=None, params={"latency": latency, "query_plan": str(plan)})
        return {"status": "success", "latency": latency, "query_plan": plan}
    except sqlite3.OperationalError as e:
        error_logger.log_error("performance_db", str(e), str(e.__traceback__), sql_statement="INSERT INTO errors", sql_error_code=e.sqlite_errorcode, params=None)
        logger.error(f"Performance test failed due to DB lock: {str(e)}")
        raise HTTPException(status_code=429, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": "Database lock", "data": str(e)}})
    except Exception as e:
        error_logger.log_error("performance", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Performance analysis failed: {str(e)}")
        raise HTTPException(status_code=400, detail={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}})

# xAI Artifact Tags: #vial2 #monitoring #performance #sqlite #neon_mcp
