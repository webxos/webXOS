import logging
from fastapi import FastAPI, HTTPException
from .db.db_manager import DatabaseManager
import redis
import os

logger = logging.getLogger(__name__)

def add_health_check(app: FastAPI):
    """Add health check endpoints to the FastAPI application.

    Args:
        app (FastAPI): FastAPI application instance.
    """
    postgres_config = {
        "host": os.getenv("POSTGRES_HOST", "postgresdb"),
        "port": int(os.getenv("POSTGRES_DOCKER_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "database": os.getenv("POSTGRES_DB", "vial_mcp")
    }
    mysql_config = {
        "host": os.getenv("MYSQL_HOST", "mysqldb"),
        "port": int(os.getenv("MYSQL_DOCKER_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_ROOT_PASSWORD", "mysql"),
        "database": os.getenv("MYSQL_DB", "vial_mcp")
    }
    mongo_config = {
        "host": os.getenv("MONGO_HOST", "mongodb"),
        "port": int(os.getenv("MONGO_DOCKER_PORT", 27017)),
        "username": os.getenv("MONGO_USER", "mongo"),
        "password": os.getenv("MONGO_PASSWORD", "mongo")
    }
    redis_config = {
        "host": os.getenv("REDIS_HOST", "redis"),
        "port": int(os.getenv("REDIS_PORT", 6379))
    }

    db_manager = DatabaseManager(postgres_config, mysql_config, mongo_config)

    @app.get("/health")
    async def health_check():
        """Check the health of all services.

        Returns:
            dict: Health status of services.

        Raises:
            HTTPException: If any service is unhealthy.
        """
        try:
            status = {"services": {}}

            # Check PostgreSQL
            try:
                db_manager.connect_postgres()
                with db_manager.postgres_conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                status["services"]["postgres"] = "healthy"
            except Exception as e:
                status["services"]["postgres"] = f"unhealthy: {str(e)}"

            # Check MySQL
            try:
                db_manager.connect_mysql()
                with db_manager.mysql_conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                status["services"]["mysql"] = "healthy"
            except Exception as e:
                status["services"]["mysql"] = f"unhealthy: {str(e)}"

            # Check MongoDB
            try:
                db_manager.connect_mongo()
                db_manager.mongo_client.admin.command("ping")
                status["services"]["mongo"] = "healthy"
            except Exception as e:
                status["services"]["mongo"] = f"unhealthy: {str(e)}"

            # Check Redis
            try:
                redis_client = redis.Redis(**redis_config, decode_responses=True)
                redis_client.ping()
                status["services"]["redis"] = "healthy"
            except Exception as e:
                status["services"]["redis"] = f"unhealthy: {str(e)}"

            # Overall status
            status["overall"] = "healthy" if all("healthy" in v for v in status["services"].values()) else "unhealthy"
            logger.info(f"Health check: {status}")

            if status["overall"] == "unhealthy":
                raise HTTPException(status_code=503, detail=status)
            return status
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [HealthCheck] Health check failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
