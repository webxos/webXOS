import redis
import logging
import datetime
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

class MonitorAgent:
    def __init__(self):
        self.metrics_key = "api_metrics"

    async def log_api_call(self, endpoint: str, user_id: str, status_code: int, latency: float):
        try:
            metric = {
                "endpoint": endpoint,
                "user_id": user_id,
                "status_code": status_code,
                "latency": latency,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            redis_client.lpush(self.metrics_key, json.dumps(metric))
            redis_client.ltrim(self.metrics_key, 0, 999)  # Keep last 1000 metrics
        except Exception as e:
            logger.error(f"API logging error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** API logging error: {str(e)}\n")

    async def get_metrics(self) -> List[Dict[str, Any]]:
        try:
            metrics = redis_client.lrange(self.metrics_key, 0, -1)
            return [json.loads(m) for m in metrics]
        except Exception as e:
            logger.error(f"Metrics retrieval error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Metrics retrieval error: {str(e)}\n")
            return []
