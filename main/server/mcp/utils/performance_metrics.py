# main/server/mcp/utils/performance_metrics.py
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge
from ..utils.mcp_error_handler import MCPError
import logging
import time
import os

logger = logging.getLogger("mcp")

class PerformanceMetrics:
    def __init__(self):
        self.requests_total = Counter("mcp_requests_total", "Total MCP requests", ["endpoint"])
        self.request_duration = Histogram("mcp_request_duration_seconds", "Request duration", ["endpoint"])
        self.error_rate = Counter("mcp_errors_total", "Total errors", ["endpoint", "error_code"])
        self.sla_compliance = Gauge("mcp_sla_compliance", "SLA compliance percentage")
        self.active_users = Gauge("mcp_active_users", "Number of active users")
        self.uptime_start = time.time()

    def track_request(self, endpoint: str):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    self.requests_total.labels(endpoint=endpoint).inc()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.request_duration.labels(endpoint=endpoint).observe(duration)
                    
                    # Update SLA compliance (target: 99.9% uptime)
                    uptime = (time.time() - self.uptime_start) / 3600
                    downtime = max(0, duration - 0.001)  # Assume 0.1% max downtime
                    self.sla_compliance.set(100 * (1 - downtime / uptime))
                    return result
                except MCPError as e:
                    self.error_rate.labels(endpoint=endpoint, error_code=e.code).inc()
                    raise
                except Exception as e:
                    self.error_rate.labels(endpoint=endpoint, error_code=-32603).inc()
                    raise MCPError(code=-32603, message=f"Internal error: {str(e)}")
            return wrapper
        return decorator

    def update_active_users(self, count: int):
        try:
            self.active_users.set(count)
            logger.info(f"Updated active users: {count}")
        except Exception as e:
            logger.error(f"Failed to update active users: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to update active users: {str(e)}")

    async def get_metrics(self) -> Dict[str, Any]:
        try:
            metrics = {
                "requests_total": {k: v for k, v in self.requests_total._metrics.items()},
                "request_duration": {k: v for k, v in self.request_duration._metrics.items()},
                "error_rate": {k: v for k, v in self.error_rate._metrics.items()},
                "sla_compliance": self.sla_compliance._value.get(),
                "active_users": self.active_users._value.get(),
                "uptime_hours": (time.time() - self.uptime_start) / 3600
            }
            logger.info("Retrieved performance metrics")
            return metrics
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to retrieve metrics: {str(e)}")
