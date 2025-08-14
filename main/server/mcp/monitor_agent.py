import logging
import os
import psutil
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class MonitorAgent:
    """Monitors server performance metrics for Vial MCP."""
    def __init__(self, log_file: str = "/app/performance_metrics.log"):
        """Initialize MonitorAgent with log file path.

        Args:
            log_file (str): Path to performance metrics log.
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        logger.info("MonitorAgent initialized")

    def log_metrics(self) -> dict:
        """Log CPU, memory, and disk usage metrics.

        Returns:
            dict: Current performance metrics.

        Raises:
            HTTPException: If metric logging fails.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_usage_percent": cpu_usage,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_total_mb": memory.total / 1024 / 1024,
                "disk_used_gb": disk.used / 1024 / 1024 / 1024,
                "disk_total_gb": disk.total / 1024 / 1024 / 1024
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            logger.info(f"Logged performance metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Metrics logging failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [MonitorAgent] Metrics logging failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Metrics logging failed: {str(e)}")

    def get_metrics(self, limit: int = 10) -> list:
        """Retrieve recent performance metrics.

        Args:
            limit (int): Maximum number of metric entries to retrieve.

        Returns:
            list: List of recent metrics.

        Raises:
            HTTPException: If metrics retrieval fails.
        """
        try:
            metrics = []
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    lines = f.readlines()[-limit:]
                    metrics = [json.loads(line.strip()) for line in lines]
            logger.info(f"Retrieved {len(metrics)} performance metrics")
            return metrics
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [MonitorAgent] Metrics retrieval failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")
