from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import psutil
import time

logger = logging.getLogger(__name__)

async def check_health():
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        query = "INSERT INTO system_health (timestamp, cpu_usage, memory_usage) VALUES ($1, $2, $3)"
        await neon_db.execute(query, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cpu_usage, memory)
        logger.info(f"Health check: CPU {cpu_usage}%, Memory {memory}%")
        return {"status": "healthy" if cpu_usage < 80 and memory < 90 else "unhealthy"}
    except Exception as e:
        error_logger.log_error("health_monitor", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Health monitor failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #monitoring #health #neon_mcp
