import asyncpg
from config.config import DatabaseConfig
from . import 0001_initial
import logging

logger = logging.getLogger(__name__)

class MigrationManager:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.migrations = [
            ("0001_initial", 0001_initial.Migration)
        ]

    async def apply_migrations(self):
        try:
            await self.db.query("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            applied = await self.db.query("SELECT name FROM migrations")
            applied_names = {row["name"] for row in applied}
            for name, migration_class in self.migrations:
                if name not in applied_names:
                    migration = migration_class(self.db)
                    await migration.up()
                    logger.info(f"Applied migration {name} [migrate.py:25] [ID:migration_applied]")
        except Exception as e:
            logger.error(f"Migration application failed: {str(e)} [migrate.py:30] [ID:migration_error]")
            raise

    async def rollback_migration(self, name: str):
        try:
            for migration_name, migration_class in reversed(self.migrations):
                if migration_name == name:
                    migration = migration_class(self.db)
                    await migration.down()
                    await self.db.query("DELETE FROM migrations WHERE name = $1", [name])
                    logger.info(f"Rolled back migration {name} [migrate.py:35] [ID:rollback_success]")
                    return
            logger.error(f"Migration {name} not found [migrate.py:40] [ID:migration_not_found]")
            raise ValueError(f"Migration {name} not found")
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)} [migrate.py:45] [ID:rollback_error]")
            raise
