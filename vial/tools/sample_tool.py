import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleTool:
    def process_task(self, task: str) -> str:
        try:
            result = f"Processed: {task.upper()}"
            logger.info(f"Sample tool processed task: {result}")
            return result
        except Exception as e:
            logger.error(f"Sample tool error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Sample tool error: {str(e)}\n")
            raise
