from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class ErrorAggregator:
    async def aggregate_errors(self, vial_id: str):
        try:
            query = "SELECT COUNT(*) FROM vial_logs WHERE vial_id = $1 AND event_type = 'error' AND created_at > NOW() - INTERVAL '24 hours'"
            count = await neon_db.execute(query, vial_id)
            logger.info(f"Aggregated {count} errors for vial {vial_id}")
            return {"error_count": count, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        except Exception as e:
            error_logger.log_error("error_aggregate", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Error aggregation failed: {str(e)}")
            raise

error_aggregator = ErrorAggregator()

# xAI Artifact Tags: #vial2 #mcp #error #aggregator #neon #neon_mcp
