import asyncpg
from mcp.error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class NeonDBConnection:
    _pool = None

    @classmethod
    async def connect(cls):
        try:
            cls._pool = await asyncpg.create_pool(os.getenv("NEON_DB_URL"))
            logger.info("NeonDB connection pool created")
        except Exception as e:
            error_logger.log_error("neon_connect", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"NeonDB connection failed: {str(e)}")
            raise

    @classmethod
    async def disconnect(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("NeonDB connection pool closed")

    @classmethod
    async def execute(cls, query, *args):
        async with cls._pool.acquire() as connection:
            return await connection.execute(query, *args)

neon_db = NeonDBConnection()

# xAI Artifact Tags: #vial2 #mcp #database #neon #connection #neon_mcp
