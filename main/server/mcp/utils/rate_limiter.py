# main/server/mcp/utils/rate_limiter.py
import redis
import time
from typing import Optional
from fastapi import HTTPException
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
import os

class RateLimiter:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.redis_client = None
        self.limit = int(os.getenv("RATE_LIMIT", 100))  # Requests per window
        self.window = int(os.getenv("RATE_WINDOW_SECONDS", 60))  # Time window in seconds
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
        except redis.ConnectionError:
            self.redis_client = None
            self.in_memory_cache = {}

    def check_rate_limit(self, key: str) -> bool:
        with self.metrics.track_span("check_rate_limit", {"key": key}):
            try:
                current_time = int(time.time())
                window_start = current_time - (current_time % self.window)
                key_with_window = f"rate_limit:{key}:{window_start}"

                if self.redis_client:
                    count = self.redis_client.incr(key_with_window)
                    if count == 1:
                        self.redis_client.expire(key_with_window, self.window)
                    if count > self.limit:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                    return True
                else:
                    if key_with_window not in self.in_memory_cache:
                        self.in_memory_cache[key_with_window] = {"count": 0, "expiry": current_time + self.window}
                    self.in_memory_cache[key_with_window]["count"] += 1
                    if self.in_memory_cache[key_with_window]["count"] > self.limit:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                    if current_time > self.in_memory_cache[key_with_window]["expiry"]:
                        del self.in_memory_cache[key_with_window]
                    return True
            except Exception as e:
                handle_generic_error(e, context="check_rate_limit")
                raise

    def reset_rate_limit(self, key: str) -> bool:
        with self.metrics.track_span("reset_rate_limit", {"key": key}):
            try:
                if self.redis_client:
                    keys = self.redis_client.keys(f"rate_limit:{key}:*")
                    if keys:
                        self.redis_client.delete(*keys)
                    return True
                else:
                    keys = [k for k in self.in_memory_cache.keys() if k.startswith(f"rate_limit:{key}:")]
                    for k in keys:
                        del self.in_memory_cache[k]
                    return True
            except Exception as e:
                handle_generic_error(e, context="reset_rate_limit")
                return False
