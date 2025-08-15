# main/server/mcp/utils/cache_manager.py
import redis
import json
import time
from typing import Any, Optional
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
import os

class CacheManager:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.redis_client = None
        self.in_memory_cache = {}
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
        except redis.ConnectionError:
            self.redis_client = None

    def set(self, key: str, value: Any, expiry: int = 3600) -> bool:
        with self.metrics.track_span("cache_set", {"key": key}):
            try:
                serialized_value = json.dumps(value)
                if self.redis_client:
                    self.redis_client.setex(key, expiry, serialized_value)
                    return True
                else:
                    self.in_memory_cache[key] = {"value": value, "expiry": int(time.time()) + expiry}
                    return True
            except Exception as e:
                handle_generic_error(e, context="cache_set")
                return False

    def get(self, key: str) -> Optional[Any]:
        with self.metrics.track_span("cache_get", {"key": key}):
            try:
                if self.redis_client:
                    value = self.redis_client.get(key)
                    return json.loads(value) if value else None
                else:
                    if key in self.in_memory_cache:
                        if time.time() < self.in_memory_cache[key]["expiry"]:
                            return self.in_memory_cache[key]["value"]
                        else:
                            del self.in_memory_cache[key]
                    return None
            except Exception as e:
                handle_generic_error(e, context="cache_get")
                return None

    def delete(self, key: str) -> bool:
        with self.metrics.track_span("cache_delete", {"key": key}):
            try:
                if self.redis_client:
                    return self.redis_client.delete(key) > 0
                else:
                    if key in self.in_memory_cache:
                        del self.in_memory_cache[key]
                        return True
                    return False
            except Exception as e:
                handle_generic_error(e, context="cache_delete")
                return False
