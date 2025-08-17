import os
import asyncpg
import asyncio
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.conn = None
        self.url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_EzPpBWkGdm69@ep-sparkling-thunder-aetjtveu-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
        self.data_api_url = "https://app-billowing-king-08029676.dpl.myneon.app"
        self.project_id = os.getenv("NEON_PROJECT_ID", "twilight-art-21036984")

    async def connect(self):
        try:
            self.conn = await asyncpg.connect(self.url)
            logger.info(f"Connected to database: {self.url}")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise Exception(f"Database connection failed: {str(e)}")

    async def disconnect(self):
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")

    async def query(self, query: str, params: list = None):
        try:
            if query.strip().upper().startswith("SELECT"):
                result = await self.conn.fetch(query, *(params or []))
            else:
                result = await self.conn.execute(query, *(params or []))
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise Exception(f"Query failed: {str(e)}")
