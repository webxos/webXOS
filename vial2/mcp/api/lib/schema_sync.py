import asyncpg
from config.config import DatabaseConfig
import logging

logger = logging.getLogger(__name__)

class SchemaSync:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.project_id = db.project_id

    async def sync_schema(self, source_conn: str, tables: list):
        try:
            source_pool = await asyncpg.create_pool(
                dsn=source_conn,
                min_size=1,
                max_size=5,
                ssl="require",
                command_timeout=30
            )
            async with source_pool.acquire() as conn:
                for table in tables:
                    schema = await conn.fetch(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = $1 AND table_schema = 'public'
                    """, table)
                    schema_sql = f"CREATE TABLE IF NOT EXISTS {table} ("
                    columns = []
                    for col in schema:
                        nullable = "NOT NULL" if col["is_nullable"] == "NO" else ""
                        columns.append(f"{col['column_name']} {col['data_type']} {nullable}")
                    schema_sql += ", ".join(columns) + ", PRIMARY KEY (id));"
                    await self.db.query(schema_sql)
            await source_pool.close()
            logger.info(f"Schema synced for tables: {tables} [schema_sync.py:25] [ID:schema_sync_success]")
            return {"status": "success", "tables": tables}
        except Exception as e:
            error_message = f"Schema sync failed: {str(e)} [schema_sync.py:30] [ID:schema_sync_error]"
            logger.error(error_message)
            return {"error": error_message}
