import os
import logging
from fastapi import HTTPException
from typing import Dict

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates environment and database configurations for Vial MCP."""
    def __init__(self):
        """Initialize ConfigValidator."""
        logger.info("ConfigValidator initialized")

    def validate_env(self) -> None:
        """Validate required environment variables.

        Raises:
            HTTPException: If any required environment variable is missing.
        """
        required_vars = [
            "POSTGRES_HOST", "POSTGRES_DOCKER_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
            "MYSQL_HOST", "MYSQL_DOCKER_PORT", "MYSQL_USER", "MYSQL_ROOT_PASSWORD", "MYSQL_DB",
            "MONGO_HOST", "MONGO_DOCKER_PORT", "MONGO_USER", "MONGO_PASSWORD",
            "REDIS_HOST", "REDIS_PORT",
            "PYTHON_LOCAL_PORT", "PYTHON_DOCKER_PORT",
            "JWT_SECRET"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] {error_msg}\n")
            raise HTTPException(status_code=500, detail=error_msg)

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
            for config, db_name in [(postgres_config, "PostgreSQL"), (mysql_config, "MySQL"), (mongo_config, "MongoDB")]:
                required_keys = ["host", "port", "user", "password"] if db_name != "MongoDB" else ["host", "port", "username", "password"]
                if db_name != "MongoDB":
                    required_keys.append("database")
                missing_keys = [key for key in required_keys if key not in config or not config[key]]
                if missing_keys:
                    error_msg = f"Invalid {db_name} config: missing {', '.join(missing_keys)}"
                    logger.error(error_msg)
                    with open("/app/errorlog.md", "a") as f:
                        f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] {error_msg}\n")
                    raise HTTPException(status_code=500, detail=error_msg)
            logger.info("Database configurations validated successfully")
        except Exception as e:
            logger.error(f"Database config validation failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [ConfigValidator] Database config validation failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Database config validation failed: {str(e)}")
