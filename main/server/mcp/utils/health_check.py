# main/server/mcp/utils/health_check.py
from fastapi import APIRouter, HTTPException
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.mcp_error_handler import MCPError
from pymongo import MongoClient
import redis.asyncio as redis
import os
import logging
import asyncio

router = APIRouter()
logger = logging.getLogger("mcp")

class HealthCheck:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))

    async def check_db(self) -> bool:
        try:
            await asyncio.sleep(0)  # Ensure async context
            self.mongo_client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    async def check_redis(self) -> bool:
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

    @router.get("/health")
    async def health(self):
        try:
            db_status = await self.check_db()
            redis_status = await self.check_redis()
            metrics = await self.metrics.get_metrics()
            
            if not db_status or not redis_status:
                raise HTTPException(status_code=503, detail="Service Unavailable")
            
            return {
                "status": "healthy",
                "database": "ok" if db_status else "failed",
                "redis": "ok" if redis_status else "failed",
                "metrics": metrics
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

    @router.get("/health/liveness")
    async def liveness(self):
        try:
            db_status = await self.check_db()
            if not db_status:
                raise HTTPException(status_code=503, detail="Database unavailable")
            return {"status": "alive"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Liveness check failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Liveness check failed: {str(e)}")

    @router.get("/health/readiness")
    async def readiness(self):
        try:
            db_status = await self.check_db()
            redis_status = await self.check_redis()
            if not db_status or not redis_status:
                raise HTTPException(status_code=503, detail="Service not ready")
            return {"status": "ready"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Readiness check failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")

    async def close(self):
        self.mongo_client.close()
        await self.redis_client.aclose()
