import logging
from datetime import datetime
from fastapi import HTTPException
from .auth_manager import AuthManager
from .db.db_manager import DatabaseManager
import os

logger = logging.getLogger(__name__)

class LibraryAgent:
    """Manages library operations for vector-based data processing in Vial MCP."""
    def __init__(self):
        """Initialize LibraryAgent with database and auth manager."""
        self.auth_manager = AuthManager()
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
        logger.info("LibraryAgent initialized")

    async def process_library(self, library_id: str, wallet_id: str, content: str, db_type: str, access_token: str) -> dict:
        """Process and store library data with vector-based operations.

        Args:
            library_id (str): Library ID to process.
            wallet_id (str): Wallet ID for access control.
            content (str): Library content to process.
            db_type (str): Database type (postgres, mysql, mongo).
            access_token (str): JWT access token.

        Returns:
            dict: Processing status and library ID.

        Raises:
            HTTPException: If processing or authentication fails.
        """
        try:
            payload = self.auth_manager.verify_token(access_token)
            if payload["wallet_id"] != wallet_id:
                logger.warning(f"Unauthorized wallet access: {wallet_id}")
                raise HTTPException(status_code=401, detail="Unauthorized wallet access")

            # Simple vector processing placeholder (e.g., hash for uniqueness)
            vector_id = f"vector_{hash(content) % 10000}"

            # Store in specified database
            if db_type == "postgres":
                with self.db_manager.postgres_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO libraries (library_id, wallet_id, content, vector_id, timestamp) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING library_id",
                        (library_id, wallet_id, content, vector_id, datetime.now())
                    )
                    result_id = cursor.fetchone()[0]
                    self.db_manager.postgres_conn.commit()
            elif db_type == "mysql":
                with self.db_manager.mysql_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO libraries (library_id, wallet_id, content, vector_id, timestamp) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (library_id, wallet_id, content, vector_id, datetime.now())
                    )
                    result_id = library_id
                    self.db_manager.mysql_conn.commit()
            elif db_type == "mongo":
                library_data = {
                    "library_id": library_id,
                    "wallet_id": wallet_id,
                    "content": content,
                    "vector_id": vector_id,
                    "timestamp": datetime.now()
                }
                result = self.db_manager.mongo_client.vial_mcp.libraries.insert_one(library_data)
                result_id = library_id
            else:
                raise ValueError("Invalid database type")

            logger.info(f"Processed library {library_id} for wallet {wallet_id} in {db_type}")
            return {"status": "success", "library_id": result_id, "vector_id": vector_id}
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Library processing failed for {library_id}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [LibraryAgent] Library processing failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Library processing failed: {str(e)}")
