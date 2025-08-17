from fastapi import APIRouter
from ..utils.logging import log_error, log_info
import pymongo
import redis

router = APIRouter()

async def check_database():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017")
        client.admin.command('ping')
        return {"status": "healthy", "database": "MongoDB"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_redis():
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return {"status": "healthy", "cache": "Redis"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@router.get("/health")
async def health_check():
    try:
        db_status = await check_database()
        redis_status = await check_redis()
        log_info("Health check successful")
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "webxos-mcp-gateway",
            "version": "2.7.8",
            "balance": 0.0000,
            "reputation": 0,
            "checks": {
                "database": db_status,
                "cache": redis_status
            }
        }
    except Exception as e:
        log_error(f"Traceback: Health check failed: {str(e)}")
        return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat(), "error": str(e)}, 503

@router.get("/ready")
async def ready_check():
    try:
        log_info("Ready check successful")
        return {"status": "ready"}
    except Exception as e:
        log_error(f"Traceback: Ready check failed: {str(e)}")
        return {"status": "not ready", "error": str(e)}, 503

@router.get("/live")
async def live_check():
    try:
        log_info("Live check successful")
        return {"status": "alive"}
    except Exception as e:
        log_error(f"Traceback: Live check failed: {str(e)}")
        return {"status": "not alive", "error": str(e)}, 503
