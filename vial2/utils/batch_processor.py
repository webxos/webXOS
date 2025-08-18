from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.tasks = []

    async def add_task(self, task_func, *args):
        try:
            self.tasks.append((task_func, args))
            if len(self.tasks) >= self.batch_size:
                await self.process_batch()
        except Exception as e:
            error_logger.log_error("batch_processor", str(e), str(e.__traceback__))
            logger.error(f"Batch task addition failed: {str(e)}")
            raise

    async def process_batch(self):
        try:
            results = await asyncio.gather(*[task_func(*args) for task_func, args in self.tasks], return_exceptions=True)
            self.tasks.clear()
            return results
        except Exception as e:
            error_logger.log_error("batch_processor", str(e), str(e.__traceback__))
            logger.error(f"Batch processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# xAI Artifact Tags: #vial2 #utils #batch_processor #neon_mcp
