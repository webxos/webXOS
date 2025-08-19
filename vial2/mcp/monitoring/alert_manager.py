from mcp.monitoring.resource_alert import check_resource_alert
from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

class AlertManager:
    async def manage_alerts(self):
        try:
            while True:
                health = await check_resource_alert()
                if health["status"] == "unhealthy":
                    query = "UPDATE system_alerts SET resolved = FALSE WHERE alert_type = 'resource'"
                    await neon_db.execute(query)
                    logger.warning("Unresolved alert managed")
                await asyncio.sleep(60)
        except Exception as e:
            error_logger.log_error("alert_manager", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
            logger.error(f"Alert management failed: {str(e)}")
            raise

alert_manager = AlertManager()

# xAI Artifact Tags: #vial2 #mcp #monitoring #alert #manager #neon_mcp
