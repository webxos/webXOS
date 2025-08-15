# main/server/mcp/utils/performance_metrics.py
from typing import Dict
from prometheus_client import Counter, Histogram, Gauge
from ..utils.mcp_error_handler import MCPError
import time

class PerformanceMetrics:
    def __init__(self):
        self.request_count = Counter('mcp_requests_total', 'Total MCP requests', ['endpoint'])
        self.request_latency = Histogram('mcp_request_latency_seconds', 'MCP request latency', ['endpoint'])
        self.error_count = Counter('mcp_errors_total', 'Total MCP errors', ['endpoint', 'error_code'])
        self.active_agents = Gauge('mcp_active_agents', 'Number of active agents', ['vial_id'])

    def record_request(self, endpoint: str, duration: float) -> None:
        try:
            self.request_count.labels(endpoint=endpoint).inc()
            self.request_latency.labels(endpoint=endpoint).observe(duration)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to record request metrics: {str(e)}")

    def record_error(self, endpoint: str, error_code: int) -> None:
        try:
            self.error_count.labels(endpoint=endpoint, error_code=error_code).inc()
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to record error metrics: {str(e)}")

    def set_active_agents(self, vial_id: str, count: int) -> None:
        try:
            self.active_agents.labels(vial_id=vial_id).set(count)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to set active agents metric: {str(e)}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "requests_total": {k: v for k, v in self.request_count._metrics.items()},
                "request_latency": {k: v._sum for k, v in self.request_latency._metrics.items()},
                "errors_total": {k: v for k, v in self.error_count._metrics.items()},
                "active_agents": {k: v._value.get() for k, v in self.active_agents._metrics.items()}
            }
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to retrieve metrics: {str(e)}")
