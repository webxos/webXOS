import time
import logging
import json
from config.config import DatabaseConfig
from fastapi import Request
import uuid

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.counters = {}
        self.timers = {}
        self.db = DatabaseConfig()

    def increment(self, metric: str):
        self.counters[metric] = self.counters.get(metric, 0) + 1
        logger.info(f"Metric incremented: {metric}={self.counters[metric]} [metrics.py:15] [ID:metric_increment]")

    async def time_query(self, query: str):
        try:
            start = time.time()
            result = await self.db.query(query)
            duration = time.time() - start
            self.timers[query] = duration
            logger.info(f"Query executed: {query[:50]} in {duration:.3f}s [metrics.py:20] [ID:query_time]")
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)} [metrics.py:25] [ID:query_error]")
            raise

    async def time_model_inference(self, vial_id: str, model: str):
        try:
            start = time.time()
            # Simulate inference (actual inference in vial_management.py)
            duration = time.time() - start
            metric = f"model_inference_{vial_id}"
            self.timers[metric] = duration
            logger.info(f"Model inference: {vial_id} in {duration:.3f}s [metrics.py:30] [ID:model_inference_time]")
            return duration
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)} [metrics.py:35] [ID:model_inference_error]")
            raise

    async def replication_metrics(self):
        try:
            result = await self.db.query("SELECT subname, received_lsn, latest_end_lsn, last_msg_receipt_time FROM pg_stat_subscription")
            metrics_data = {"replication_subscriptions": [dict(row) for row in result]}
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), "system", "replication_metrics", json.dumps(metrics_data), str(uuid.uuid4()), "twilight-art-21036984"]
            )
            logger.info(f"Replication metrics stored [metrics.py:40] [ID:replication_metrics_success]")
            return metrics_data
        except Exception as e:
            logger.error(f"Replication metrics failed: {str(e)} [metrics.py:45] [ID:replication_metrics_error]")
            return {"error": str(e)}

    async def middleware(self, request: Request, call_next):
        try:
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start
            metric = f"{request.method}_{request.url.path}_duration"
            self.timers[metric] = duration
            logger.info(f"Request processed: {metric} in {duration:.3f}s [metrics.py:50] [ID:request_time]")
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)} [metrics.py:55] [ID:request_error]")
            raise

    async def get_metrics(self):
        try:
            metrics_data = {
                "counters": self.counters,
                "timers": self.timers,
                "replication": await self.replication_metrics()
            }
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), "system", "metrics", json.dumps(metrics_data), str(uuid.uuid4()), "twilight-art-21036984"]
            )
            logger.info(f"Metrics stored [metrics.py:60] [ID:metrics_stored]")
            return metrics_data
        except Exception as e:
            error_message = f"Metrics storage failed: {str(e)} [metrics.py:65] [ID:metrics_error]"
            logger.error(error_message)
            return {"error": error_message}
