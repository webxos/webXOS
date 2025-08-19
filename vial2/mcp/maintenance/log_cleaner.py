from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class LogCleaner:
    async def clean_old_logs(self, days: int = 30):
        try:
            cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))
            query = "DELETE FROM error_logs WHERE timestamp < $1"
            await neon_db.execute(query, cutoff)
            logger.info(f"Cleaned logs older than {days} days")
        except Exception as e:
            error_logger.log_error("log_clean", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
            logger.error(f"Log cleaning failed: {str(e)}")
            raise

log_cleaner = LogCleaner()

# xAI Artifact Tags: #vial2 #mcp #maintenance #log #cleaner #neon_mcp
