import os
import logging
from typing import Dict
from fastapi import HTTPException
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration settings for Vial MCP."""
    def __init__(self):
        """Initialize ConfigValidator with required environment variables."""
        self.required_env_vars = [
            "POSTGRES_HOST", "POSTGRES_DOCKER_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
            "MYSQL_HOST", "MYSQL_DOCKER_PORT", "MYSQL_USER", "MYSQL_ROOT_PASSWORD", "MYSQL_DB",
            "MONGO_HOST", "MONGO_DOCKER_PORT", "MONGO_USER", "MONGO_PASSWORD",
            "REDIS_HOST", "REDIS_PORT", "JWT_SECRET",
            "PYTHON_LOCAL_PORT", "PYTHON_DOCKER_PORT", "NODE_LOCAL_PORT", "NODE_DOCKER_PORT",
            "JAVA_LOCAL_PORT", "JAVA_DOCKER_PORT"
        ]
        logger.info("ConfigValidator initialized")

    def validate_env(self) -> None:
        """Validate required environment variables.

        Raises:
            HTTPException: If any required environment variable is missing or invalid.
        """
        try:
            for var in self.required_env_vars:
                value = os.getenv(var)
                if not value:
                    logger.error(f"Missing environment variable: {var}")
                    with open("/app/errorlog.md", "a") as f:
                        f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Missing environment variable: {var}\n")
                    raise HTTPException(status_code=500, detail=f"Missing environment variable: {var}")
                if var in ["POSTGRES_DOCKER_PORT", "MYSQL_DOCKER_PORT", "MONGO_DOCKER_PORT", "REDIS_PORT",
                          "PYTHON_LOCAL_PORT", "PYTHON_DOCKER_PORT", "NODE_LOCAL_PORT", "NODE_DOCKER_PORT",
                          "JAVA_LOCAL_PORT", "JAVA_DOCKER_PORT"]:
                    try:
                        int(value)
                    except ValueError:
                        logger.error(f"Invalid port value for {var}: {value}")
                        with open("/app/errorlog.md", "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Invalid port value for {var}: {value}\n")
                        raise HTTPException(status_code=500, detail=f"Invalid port value for {var}: {value}")
            logger.info("Environment variables validated successfully")
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Environment validation failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Environment validation failed: {str(e)}")

    def validate_db_configs(self, postgres_config: Dict, mysql_config: Dict, mongo_config: Dict) -> None:
        """Validate database configuration dictionaries.

        Args:
            postgres_config (Dict): PostgreSQL configuration.
            mysql_config (Dict): MySQL configuration.
            mongo_config (Dict): MongoDB configuration.

        Raises:
            HTTPException: If any configuration is invalid.
        """
        try:
            for config, name in [(postgres_config, "PostgreSQL"), (mysql_config, "MySQL"), (mongo_config, "MongoDB")]:
                required_keys = ["host", "port", "user", "password"] if name != "MongoDB" else ["host", "port", "username", "password"]
                for key in required_keys:
                    if key not in config or not config[key]:
                        logger.error(f"Invalid {name} config: missing or empty {key}")
                        with open("/app/errorlog.md", "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Invalid {name} config: missing or empty {key}\n")
                        raise HTTPException(status_code=500, detail=f"Invalid {name} config: missing or empty {key}")
                if name != "MongoDB":
                    if "database" not in config or not config["database"]:
                        logger.error(f"Invalid {name} config: missing or empty database")
                        with open("/app/errorlog.md", "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Invalid {name} config: missing or empty database\n")
                        raise HTTPException(status_code=500, detail=f"Invalid {name} config: missing or empty database")
            logger.info("Database configurations validated successfully")
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Database config validation failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Database config validation failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Database config validation failed: {str(e)}")
