import asyncpg
from typing import Optional
from ...mcp.error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class NeonConnection:
    def __init__(self):
        self.pool = None
        self.db_url = os.getenv("DATABASE_URL", "postgresql://vial2:vial2@localhost:5432/vial2_dev")

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(self.db_url)
            logger.info("NeonDB connection established")
        except Exception as e:
            error_logger.log_error("neon_connect", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"NeonDB connection failed: {str(e)}")
            raise

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            logger.info("NeonDB connection closed")

    async def execute(self, query: str, *args):
        if not self.pool:
            raise ValueError("Database connection not established")
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(query, *args)
                return result
            except Exception as e:
                error_logger.log_error("neon_execute", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=args)
                logger.error(f"NeonDB query failed: {str(e)}")
                raise

neon_db = NeonConnection()

# xAI Artifact Tags: #vial2 #mcp #database #neon #connection #neon_mcp
