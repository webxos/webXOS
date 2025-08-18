from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ..error_logging.error_log import error_logger
from ..monitoring.metrics import collect_metrics
from ..utils.helpers import get_db_pool
import logging

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

async def schedule_metrics_collection():
    try:
        async with get_db_pool() as db:
            metrics = await collect_metrics(db)
            logger.info(f"Scheduled metrics collection: {metrics}")
    except Exception as e:
        error_logger.log_error("scheduler", f"Scheduled metrics collection failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Scheduled metrics collection failed: {str(e)}")
        raise

def setup_scheduler():
    try:
        scheduler.add_job(schedule_metrics_collection, "interval", minutes=5)
        scheduler.start()
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("scheduler", f"Scheduler setup failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Scheduler setup failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #utils #scheduler #neon_mcp
