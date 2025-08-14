import logging,redis
from fastapi import FastAPI
from datetime import datetime

logger=logging.getLogger(__name__)

def add_health_check(app:FastAPI,redis_host="redis",redis_port=6379):
    """Add health check endpoint to FastAPI app.

    Args:
        app (FastAPI): FastAPI application instance.
        redis_host (str): Redis host for rate limiting.
        redis_port (int): Redis port.
    """
    try:
        @app.get("/api/health")
        async def health():
            """Check server and Redis health.

            Returns:
                dict: Health status.

            Raises:
                Exception: If health check fails.
            """
            try:
                redis_client=redis.Redis(host=redis_host,port=redis_port,decode_responses=True)
                redis_client.ping()
                logger.info("Health check passed")
                return {"status":"healthy","timestamp":datetime.now().isoformat()}
            except redis.ConnectionError as e:
                logger.error(f"Redis connection failed: {str(e)}")
                with open("/app/errorlog.md","a") as f:
                    f.write(f"[{datetime.now().isoformat()}] [HealthCheck] Redis connection failed: {str(e)}\n")
                return {"status":"unhealthy","error":"Redis connection failed","timestamp":datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                with open("/app/errorlog.md","a") as f:
                    f.write(f"[{datetime.now().isoformat()}] [HealthCheck] Health check failed: {str(e)}\n")
                return {"status":"unhealthy","error":str(e),"timestamp":datetime.now().isoformat()}
        logger.info("Health check endpoint added")
    except Exception as e:
        logger.error(f"Failed to add health check endpoint: {str(e)}")
        with open("/app/errorlog.md","a") as f:
            f.write(f"[{datetime.now().isoformat()}] [HealthCheck] Failed to add health check endpoint: {str(e)}\n")
        raise Exception(f"Failed to add health check endpoint: {str(e)}")
