from asyncpg import create_pool
import logging
import os
from typing import Any, List

logger = logging.getLogger("mcp.config")
logger.setLevel(logging.INFO)

class DatabaseConfig:
    def __init__(self):
        self.pool = None
        db_url = os.getenv("NEON_DATABASE_URL")
        if not db_url:
            raise ValueError("NEON_DATABASE_URL not set")
        self.dsn = db_url

    async def connect(self):
        try:
            self.pool = await create_pool(
                dsn=self.dsn,
                max_size=20,
                min_size=1,
                max_inactive_connection_lifetime=30000,
                timeout=2
            )
            async with self.pool.acquire() as connection:
                await connection.execute("SELECT 1")  # Test connection
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    async def query(self, text: str, params: List[Any] = []) -> Any:
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetch(text, *params)
                logger.info(f"Query executed: {text}")
                return type("Result", (), {"rows": result})  # Mimic pg.Pool result
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise

    async def close(self):
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Database close error: {str(e)}")
            raise
