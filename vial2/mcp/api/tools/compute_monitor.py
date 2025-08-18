import asyncio
import logging
from config.config import DatabaseConfig
from lib.security import SecurityHandler

logger = logging.getLogger(__name__)

class ComputeMonitor:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.project_id = db.project_id

    async def monitor_activity(self, compute_id: str, user_id: str):
        try:
            while True:
                last_activity = await self.db.query(
                    "SELECT last_activity FROM computes WHERE compute_id = $1 AND project_id = $2",
                    [compute_id, self.project_id]
                )
                if last_activity:
                    await self.db.query(
                        "UPDATE computes SET last_activity = NOW() WHERE compute_id = $1 AND project_id = $2",
                        [compute_id, self.project_id]
                    )
                    await self.security.log_action(user_id, "compute_monitor", {"compute_id": compute_id, "last_activity": last_activity[0]["last_activity"]})
                logger.info(f"Monitored activity for compute: {compute_id} [compute_monitor.py:20] [ID:monitor_success]")
                await asyncio.sleep(60)
        except Exception as e:
            error_message = f"Compute monitoring failed: {str(e)} [compute_monitor.py:25] [ID:monitor_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "compute_monitor", error_message)
            return {"error": error_message}
