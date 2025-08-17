import os
import asyncpg
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.url = os.getenv("DATABASE_URL")
        self.pool = None
        self.project_id = os.getenv("NEON_PROJECT_ID", "twilight-art-21036984")
        self.data_api_url = os.getenv("DATA_API_URL", "https://app-billowing-king-08029676.dpl.myneon.app")
        self.jwt_audience = os.getenv("JWT_AUDIENCE", "vial-mcp-webxos")
        self.stack_auth_client_id = os.getenv("STACK_AUTH_CLIENT_ID")
        self.stack_auth_client_secret = os.getenv("STACK_AUTH_CLIENT_SECRET")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY")

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.url,
                min_size=1,
                max_size=10,
                server_settings={"application_name": "vial_mcp"},
                ssl="require",
                command_timeout=30
            )
            logger.info(f"Database pool created [config.py:25] [ID:pool_success]")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)} [config.py:30] [ID:pool_error]")
            raise

    async def disconnect(self):
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Database pool closed [config.py:35] [ID:pool_close_success]")
        except Exception as e:
            logger.error(f"Database pool closure failed: {str(e)} [config.py:40] [ID:pool_close_error]")
            raise

    async def query(self, query: str, args: list = None):
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetch(query, *(args or []))
                logger.info(f"Query executed: {query[:50]}... [config.py:45] [ID:query_success]")
                return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)} [config.py:50] [ID:query_error]")
            raise
