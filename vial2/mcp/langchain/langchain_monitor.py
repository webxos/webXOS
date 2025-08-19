from langchain.adapters.monitoring import MonitoringAdapter
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class LangChainMonitor:
    def __init__(self):
        self.monitor = MonitoringAdapter()

    def track_usage(self, vial_id: str, usage_data: dict):
        try:
            metrics = self.monitor.track_usage(usage_data)
            logger.info(f"Tracked usage for vial {vial_id}: {metrics}")
            return metrics
        except Exception as e:
            error_logger.log_error("langchain_monitor_track", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"LangChain monitoring failed: {str(e)}")
            raise

langchain_monitor = LangChainMonitor()

# xAI Artifact Tags: #vial2 #mcp #langchain #monitor #neon #neon_mcp
