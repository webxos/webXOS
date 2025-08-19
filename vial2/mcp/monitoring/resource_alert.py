from mcp.monitoring.health_monitor import check_health
from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

async def check_resource_alert():
    try:
        health = await check_health()
        if health["status"] == "unhealthy":
            query = "INSERT INTO system_alerts (timestamp, alert_type, details) VALUES ($1, $2, $3)"
            await neon_db.execute(query, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resource", health)
            logger.warning("Resource alert triggered")
        return health
    except Exception as e:
        error_logger.log_error("resource_alert", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Resource alert failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #monitoring #resource #alert #neon_mcp
