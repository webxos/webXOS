# main/server/mcp/utils/health_check.py
from fastapi import APIRouter
from pymongo import MongoClient
import os
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error

router = APIRouter()
metrics = PerformanceMetrics()
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))

@router.get("/health")
async def health_check():
    with metrics.track_span("health_check"):
        try:
            mongo_status = mongo_client.admin.command('ping')["ok"] == 1
            return {
                "status": "healthy" if mongo_status else "unhealthy",
                "mongo": mongo_status,
                "timestamp": metrics.datetime.utcnow().isoformat()
            }
        except Exception as e:
            handle_generic_error(e, context="health_check")
            return {
                "status": "unhealthy",
                "mongo": False,
                "timestamp": metrics.datetime.utcnow().isoformat(),
                "error": str(e)
            }
