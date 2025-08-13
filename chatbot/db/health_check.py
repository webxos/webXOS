import os
import logging
import httpx
from pymongo import MongoClient
from pymongo.errors import ConnectionError
import psycopg2
from psycopg2 import OperationalError
import redis
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(filename='/db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://user:password@localhost:5432/vial_mcp')
REDIS_URI = os.getenv('REDIS_URI', 'redis://localhost:6379')
SERVICE_URLS = {
    "API Gateway": "http://localhost:8000/api/health",
    "Authentication": "http://localhost:8001/auth/health",
    "Vial Management": "http://localhost:8002/vials/health",
    "Blockchain": "http://localhost:8003/blockchain/health",
    "Wallet": "http://localhost:8004/wallet/health",
    "Quantum Link": "http://localhost:8005/quantum/health",
    "Task Processing": "http://localhost:8006/tasks/health",
    "Monitoring": "http://localhost:8007/monitoring/health"
}

async def check_databases():
    db_status = {"mongo": False, "postgres": False, "redis": False}
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db_status["mongo"] = True
        logger.info("MongoDB health check passed")
    except ConnectionError as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
    
    try:
        conn = psycopg2.connect(POSTGRES_URI)
        conn.close()
        db_status["postgres"] = True
        logger.info("PostgreSQL health check passed")
    except OperationalError as e:
        logger.error(f"PostgreSQL health check failed: {str(e)}")
    
    try:
        r = redis.Redis.from_url(REDIS_URI, decode_responses=True)
        r.ping()
        db_status["redis"] = True
        logger.info("Redis health check passed")
    except redis.ConnectionError as e:
        logger.error(f"Redis health check failed: {str(e)}")
    
    return db_status

async def check_services():
    service_status = {}
    async with httpx.AsyncClient() as client:
        for name, url in SERVICE_URLS.items():
            try:
                response = await client.get(url, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    service_status[name] = data.get("status", "unknown")
                    logger.info(f"Health check passed for {name}: {data}")
                else:
                    service_status[name] = "down"
                    logger.error(f"Health check failed for {name}: HTTP {response.status_code}")
            except Exception as e:
                service_status[name] = "down"
                logger.error(f"Health check failed for {name}: {str(e)}")
    return service_status

async def run_health_check():
    try:
        db_status = await check_databases()
        service_status = await check_services()
        overall_status = "healthy" if all(db_status.values()) and all(s == "healthy" for s in service_status.values()) else "unhealthy"
        logger.info(f"Health check completed: Overall status: {overall_status}")
        return {
            "status": overall_status,
            "databases": db_status,
            "services": service_status
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_health_check())
    print(result)
