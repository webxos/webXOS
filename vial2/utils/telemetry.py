from collections import defaultdict
from ..error_logging.error_log import error_logger
import logging
import time

logger = logging.getLogger(__name__)

class Telemetry:
    def __init__(self):
        self.metrics = defaultdict(int)

    def record_event(self, event_type: str, data: dict):
        try:
            self.metrics[event_type] += 1
            self.metrics[f"{event_type}_data"] = data
            self.metrics[f"{event_type}_timestamp"] = time.time()
        except Exception as e:
            error_logger.log_error("telemetry", str(e), str(e.__traceback__))
            logger.error(f"Telemetry recording failed: {str(e)}")
            raise

    def get_metrics(self):
        return dict(self.metrics)

# xAI Artifact Tags: #vial2 #utils #telemetry #neon_mcp
