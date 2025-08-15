# main/server/mcp/utils/health_check.py
from fastapi import FastAPI, HTTPException
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from ..db.db_manager import DBManager
from pymongo import MongoClient
import os
import redis
from typing import Dict

app = FastAPI(title="Vial MCP Health Check")
metrics = PerformanceMetrics()
db_manager = DBManager()

class HealthStatus:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
        except redis.ConnectionError:
            self.redis_client = None

    def check_mongo(self) -> bool:
        with self.metrics.track_span("check_mongo_health"):
            try:
                self.mongo_client.admin.command("ping")
                return True
            except Exception as e:
                handle_generic_error(e, context="check_mongo_health")
                return False

    def check_redis(self) -> bool:
        with self.metrics.track_span("check_redis_health"):
            try:
                if self.redis_client:
                    self.redis_client.ping()
                    return True
                return False
            except Exception as e:
                handle_generic_error(e, context="check_redis_health")
                return False

@app.get("/health", response_model=Dict)
async def health_check():
    with metrics.track_span("health_check"):
        try:
            health_status = HealthStatus()
            status = {
                "mongo": health_status.check_mongo(),
                "redis": health_status.check_redis(),
                "timestamp": datetime.utcnow().isoformat()
            }
            overall_status = all(status[k] for k in ["mongo", "redis"] if status[k] is not None)
            status["overall"] = overall_status
            if not overall_status:
                raise HTTPException(status_code=503, detail="One or more services are unhealthy")
            return status
        except Exception as e:
            handle_generic_error(e, context="health_check")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
