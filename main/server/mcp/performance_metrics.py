import logging
import time
from datetime import datetime
from fastapi import HTTPException
import os
import json

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Collects and analyzes performance metrics for Vial MCP API endpoints."""
    def __init__(self, metrics_file: str = "/app/performance_metrics.jsonl"):
        """Initialize PerformanceMetrics with metrics file path.

        Args:
            metrics_file (str): Path to store performance metrics.
        """
        self.metrics_file = metrics_file
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        logger.info("PerformanceMetrics initialized")

    def record_endpoint_metrics(self, endpoint: str, wallet_id: str, response_time: float, status_code: int) -> None:
        """Record metrics for an API endpoint call.

        Args:
            endpoint (str): API endpoint (e.g., '/api/notes/add').
            wallet_id (str): Wallet ID making the request.
            response_time (float): Response time in seconds.
            status_code (int): HTTP status code of the response.

        Raises:
            HTTPException: If recording fails.
        """
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "wallet_id": wallet_id,
                "response_time": response_time,
                "status_code": status_code
            }
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            logger.info(f"Recorded metrics for {endpoint}: {metrics}")
        except Exception as e:
            logger.error(f"Metrics recording failed for {endpoint}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [PerformanceMetrics] Metrics recording failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Metrics recording failed: {str(e)}")

    def get_endpoint_metrics(self, endpoint: str = None, limit: int = 100) -> list:
        """Retrieve performance metrics, optionally filtered by endpoint.

        Args:
            endpoint (str, optional): Filter metrics by endpoint.
            limit (int): Maximum number of metric entries to retrieve.

        Returns:
            list: List of performance metrics.

        Raises:
            HTTPException: If retrieval fails.
        """
        try:
            metrics = []
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, "r") as f:
                    lines = f.readlines()[-limit:]
                    for line in lines:
                        metric = json.loads(line.strip())
                        if endpoint is None or metric["endpoint"] == endpoint:
                            metrics.append(metric)
            logger.info(f"Retrieved {len(metrics)} metrics for endpoint {endpoint or 'all'}")
            return metrics
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [PerformanceMetrics] Metrics retrieval failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")
