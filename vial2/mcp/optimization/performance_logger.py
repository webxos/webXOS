from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class PerformanceLogger:
    async def log_performance(self, vial_id: str, duration: float):
        try:
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "performance_log", {"duration": duration, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            logger.info(f"Logged performance for vial {vial_id}: {duration}s")
        except Exception as e:
            error_logger.log_error("performance_log", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Performance logging failed: {str(e)}")
            raise

performance_logger = PerformanceLogger()

# xAI Artifact Tags: #vial2 #mcp #optimization #performance #logger #neon_mcp
