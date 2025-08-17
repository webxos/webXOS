import asyncpg
from config.config import DatabaseConfig
from lib.security import SecurityHandler
from lib.schema_sync import SchemaSync
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class ReplicationTool:
    def __init__(self, db: DatabaseConfig, security: SecurityHandler):
        self.db = db
        self.security = security
        self.project_id = db.project_id
        self.schema_sync = SchemaSync(db)

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            user_id = data.get("user_id")
            project_id = data.get("project_id", self.project_id)
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [replication_tool.py:20] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            if method == "setup_publication":
                return await self.setup_publication(user_id, data.get("tables", []), project_id)
            elif method == "setup_subscription":
                return await self.setup_subscription(user_id, data.get("source_conn"), data.get("publication_name"), project_id)
            elif method == "verify_replication":
                return await self.verify_replication(user_id, data.get("subscription_name"), project_id)
            else:
                error_message = f"Invalid replication method: {method} [replication_tool.py:25] [ID:replication_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Replication operation failed: {str(e)} [replication_tool.py:30] [ID:replication_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "replication", error_message)
            return {"error": error_message}

    async def setup_publication(self, user_id: str, tables: list, project_id: str) -> dict:
        try:
            publication_name = f"pub_{uuid.uuid4().hex[:8]}"
            table_list = ", ".join(tables) if tables else "ALL TABLES"
            await self.db.query(f"CREATE PUBLICATION {publication_name} FOR {table_list}")
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "publication", json.dumps({"name": publication_name, "tables": tables}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Publication created: {publication_name} [replication_tool.py:35] [ID:publication_success]")
            return {"status": "success", "publication_name": publication_name}
        except Exception as e:
            error_message = f"Publication setup failed: {str(e)} [replication_tool.py:40] [ID:publication_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def setup_subscription(self, user_id: str, source_conn: str, publication_name: str, project_id: str) -> dict:
        try:
            subscription_name = f"sub_{uuid.uuid4().hex[:8]}"
            await self.schema_sync.sync_schema(source_conn, ["playing_with_neon"])
            await self.db.query(
                f"CREATE SUBSCRIPTION {subscription_name} CONNECTION '{source_conn}' PUBLICATION {publication_name}"
            )
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "subscription", json.dumps({"name": subscription_name, "publication": publication_name}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Subscription created: {subscription_name} [replication_tool.py:45] [ID:subscription_success]")
            return {"status": "success", "subscription_name": subscription_name}
        except Exception as e:
            error_message = f"Subscription setup failed: {str(e)} [replication_tool.py:50] [ID:subscription_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def verify_replication(self, user_id: str, subscription_name: str, project_id: str) -> dict:
        try:
            result = await self.db.query(f"SELECT subname, received_lsn, latest_end_lsn, last_msg_receipt_time FROM pg_stat_subscription WHERE subname = $1", [subscription_name])
            if not result:
                error_message = f"Subscription not found: {subscription_name} [replication_tool.py:55] [ID:subscription_not_found]"
                logger.error(error_message)
                return {"error": error_message}
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "verify_replication", json.dumps(result[0]), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Replication verified: {subscription_name} [replication_tool.py:60] [ID:verify_replication_success]")
            return {"status": "success", "details": result[0]}
        except Exception as e:
            error_message = f"Replication verification failed: {str(e)} [replication_tool.py:65] [ID:verify_replication_error]"
            logger.error(error_message)
            return {"error": error_message}
