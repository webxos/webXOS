# main/server/mcp/utils/health_check.py
from fastapi import APIRouter
from pymongo import MongoClient
import redis.asyncio as redis
from ..utils.mcp_error_handler import MCPError
import os

router = APIRouter()

class HealthStatus:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))

    async def check_mongodb(self) -> bool:
        try:
            self.mongo_client.admin.command("ping")
            return True
        except Exception:
            return False

    async def check_redis(self) -> bool:
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False

    async def get_health(self) -> dict:
        try:
            mongo_status = await self.check_mongodb()
            redis_status = await self.check_redis()
            status = "healthy" if mongo_status and redis_status else "unhealthy"
            return {
                "status": status,
                "services": {
                    "mongodb": "up" if mongo_status else "down",
                    "redis": "up" if redis_status else "down"
                }
            }
        except Exception as e:
            raise MCPError(code=-32603, message=f"Health check failed: {str(e)}")

    async def get_readiness(self) -> dict:
        try:
            health = await self.get_health()
            return {
                "status": "ready" if health["status"] == "healthy" else "not ready",
                "services": health["services"]
            }
        except Exception as e:
            raise MCPError(code=-32603, message=f"Readiness check failed: {str(e)}")

    def close(self):
        self.mongo_client.close()

health_status = HealthStatus()

@router.get("/health")
async def health_check():
    return await health_status.get_health()

@router.get("/ready")
async def readiness_check():
    return await health_status.get_readiness()
