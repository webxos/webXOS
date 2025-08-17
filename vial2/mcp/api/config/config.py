import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_EzPpBWkGdm69@ep-sparkling-thunder-aetjtveu-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
        self.data_api_url = os.getenv("DATA_API_URL", "https://app-billowing-king-08029676.dpl.myneon.app")
        self.project_id = os.getenv("NEON_PROJECT_ID", "twilight-art-21036984")
        self.stack_auth_client_id = os.getenv("STACK_AUTH_CLIENT_ID")
        self.stack_auth_client_secret = os.getenv("STACK_AUTH_CLIENT_SECRET")
        self.stack_auth_project_id = "142ad169-5d57-4be3-bf41-6f3cd0a9ae1d"
        self.jwks_url = "https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/.well-known/jwks.json"
        self.jwt_audience = os.getenv("JWT_AUDIENCE", "vial-mcp-webxos")
        self.pool = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.url,
                min_size=1,
                max_size=10,
                ssl="require",
                command_timeout=30
            )
        except Exception as e:
            error_message = f"Database connection failed: {str(e)} [config.py:20] [ID:db_connect_error]"
            raise Exception(error_message)

    async def query(self, query: str, args: list = None):
        try:
            async with self.pool.acquire() as connection:
                if args:
                    return await connection.fetch(query, *args)
                return await connection.fetch(query)
        except Exception as e:
            error_message = f"Query failed: {str(e)} [config.py:25] [ID:query_error]"
            raise Exception(error_message)

    async def disconnect(self):
        try:
            if self.pool:
                await self.pool.close()
        except Exception as e:
            error_message = f"Database disconnection failed: {str(e)} [config.py:30] [ID:db_disconnect_error]"
            raise Exception(error_message)
