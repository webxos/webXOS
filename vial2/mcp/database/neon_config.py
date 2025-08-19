import os
from typing import Dict
from .neon_connection import neon_db
from ...mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class NeonConfig:
    def __init__(self):
        self.config = {
            "database_url": os.getenv("DATABASE_URL", "postgresql://vial2:vial2@localhost:5432/vial2_dev"),
            "neon_api_key": os.getenv("NEON_API_KEY", ""),
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10"))
        }

    async def initialize(self):
        try:
            neon_db.pool = await asyncpg.create_pool(
                dsn=self.config["database_url"],
                min_size=1,
                max_size=self.config["pool_size"],
                command_timeout=60
            )
            logger.info("NeonDB configuration initialized")
        except Exception as e:
            error_logger.log_error("neon_config_init", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=self.config)
            logger.error(f"NeonDB configuration failed: {str(e)}")
            raise

neon_config = NeonConfig()

# xAI Artifact Tags: #vial2 #mcp #database #neon #config #neon_mcp
