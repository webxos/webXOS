# main/server/mcp/utils/rate_limiter.py
from time import time
from typing import Dict
from ..utils.performance_metrics import PerformanceMetrics

class RateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests: Dict[str, list[float]] = {}
        self.metrics = PerformanceMetrics()

    def allow(self, client_id: str = "default") -> bool:
        with self.metrics.track_span("rate_limiter_check", {"client_id": client_id}):
            current_time = time()
            if client_id not in self.requests:
                self.requests[client_id] = []
            self.requests[client_id] = [t for t in self.requests[client_id] if current_time - t < self.window]
            if len(self.requests[client_id]) >= self.limit:
                return False
            self.requests[client_id].append(current_time)
            return True

    def get_remaining(self, client_id: str = "default") -> int:
        with self.metrics.track_span("rate_limiter_remaining", {"client_id": client_id}):
            current_time = time()
            if client_id not in self.requests:
                return self.limit
            self.requests[client_id] = [t for t in self.requests[client_id] if current_time - t < self.window]
            return self.limit - len(self.requests[client_id])
