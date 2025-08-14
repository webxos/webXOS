import logging
import os
from datetime import datetime
from fastapi import HTTPException
from .db.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class LibrarySync:
    """Synchronizes library data across Vial MCP databases."""
    def __init__(self):
        """Initialize LibrarySync with database manager."""
        postgres_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresdb"),
            "port": int(os.getenv("POSTGRES_DOCKER_PORT", 5432)),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "database": os.getenv("POSTGRES_DB", "vial_mcp")
        }
        mysql_config = {
            "host": os.getenv("MYSQL_HOST", "mysqldb"),
            "port": int(os.getenv("MYSQL_DOCKER_PORT", 3306)),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_ROOT_PASSWORD", "mysql"),
            "database": os.getenv("MYSQL_DB", "vial_mcp")
        }
        mongo_config = {
            "host": os.getenv("MONGO_HOST", "mongodb"),
            "port": int(os.getenv("MONGO_DOCKER_PORT", 27017)),
            "username": os.getenv("MONGO_USER", "mongo"),
            "password": os.getenv("MONGO_PASSWORD", "mongo")
        }
        self.db_manager = DatabaseManager(postgres_config, mysql_config, mongo_config)
        self.db_manager.connect()
        logger.info("LibrarySync initialized")

    async def sync_library(self, library_id: str, wallet_id: str) -> dict:
        """Synchronize library data across all databases.

        Args:
            library_id (str): Library ID to synchronize.
            wallet_id (str): Wallet ID for access control.

        Returns:
            dict: Synchronization status and details.

        Raises:
            HTTPException: If synchronization fails.
        """
        try:
            # Fetch library data from primary source (MongoDB as default)
            mongo_data = self.db_manager.mongo_client.vial_mcp.libraries.find_one({"library_id": library_id, "wallet_id": wallet_id})
            if not mongo_data:
                logger.warning(f"Library {library_id} not found for wallet {wallet_id}")
                raise HTTPException(status_code=404, detail="Library not found")

            library_data = {
                "library_id": library_id,
                "wallet_id": wallet_id,
                "content": mongo_data.get("content", ""),
                "timestamp": mongo_data.get("timestamp", datetime.now())
            }

            # Sync to PostgreSQL
            with self.db_manager.postgres_conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO libraries (library_id, wallet_id, content, timestamp) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (library_id, wallet_id) "
                    "DO UPDATE SET content = %s, timestamp = %s",
                    (library_id, wallet_id, library_data["content"], library_data["timestamp"],
                     library_data["content"], library_data["timestamp"])
                )
                self.db_manager.postgres_conn.commit()

            # Sync to MySQL
            with self.db_manager.mysql_conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO libraries (library_id, wallet_id, content, timestamp) "
                    "VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                    "content = %s, timestamp = %s",
                    (library_id, wallet_id, library_data["content"], library_data["timestamp"],
                     library_data["content"], library_data["timestamp"])
                )
                self.db_manager.mysql_conn.commit()

            logger.info(f"Synchronized library {library_id} for wallet {wallet_id}")
            return {"status": "success", "library_id": library_id, "wallet_id": wallet_id}
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Library sync failed for {library_id}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [LibrarySync] Library sync failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Library sync failed: {str(e)}")
