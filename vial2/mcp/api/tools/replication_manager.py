from config.config import DatabaseConfig
from lib.security import SecurityHandler
from tools.replication_tool import ReplicationTool
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class ReplicationManager:
    def __init__(self, db: DatabaseConfig, security: SecurityHandler):
        self.db = db
        self.security = security
        self.replication_tool = ReplicationTool(db, security)
        self.project_id = db.project_id

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            user_id = data.get("user_id")
            project_id = data.get("project_id", self.project_id)
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [replication_manager.py:20] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            if method == "pause_subscription":
                return await self.pause_subscription(user_id, data.get("subscription_name"), project_id)
            elif method == "resume_subscription":
                return await self.resume_subscription(user_id, data.get("subscription_name"), project_id)
            elif method == "drop_subscription":
                return await self.drop_subscription(user_id, data.get("subscription_name"), project_id)
            else:
                error_message = f"Invalid replication method: {method} [replication_manager.py:25] [ID:replication_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Replication operation failed: {str(e)} [replication_manager.py:30] [ID:replication_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "replication_manager", error_message)
            return {"error": error_message}

    async def pause_subscription(self, user_id: str, subscription_name: str, project_id: str) -> dict:
        try:
            await self.db.query(f"ALTER SUBSCRIPTION {subscription_name} DISABLE")
            await self.security.log_action(user_id, "pause_subscription", {"subscription_name": subscription_name})
            logger.info(f"Subscription paused: {subscription_name} [replication_manager.py:35] [ID:pause_subscription_success]")
            return {"status": "success", "subscription_name": subscription_name}
        except Exception as e:
            error_message = f"Pause subscription failed: {str(e)} [replication_manager.py:40] [ID:pause_subscription_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def resume_subscription(self, user_id: str, subscription_name: str, project_id: str) -> dict:
        try:
            await self.db.query(f"ALTER SUBSCRIPTION {subscription_name} ENABLE")
            await self.security.log_action(user_id, "resume_subscription", {"subscription_name": subscription_name})
            logger.info(f"Subscription resumed: {subscription_name} [replication_manager.py:45] [ID:resume_subscription_success]")
            return {"status": "success", "subscription_name": subscription_name}
        except Exception as e:
            error_message = f"Resume subscription failed: {str(e)} [replication_manager.py:50] [ID:resume_subscription_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def drop_subscription(self, user_id: str, subscription_name: str, project_id: str) -> dict:
        try:
            await self.db.query(f"DROP SUBSCRIPTION {subscription_name}")
            await self.security.log_action(user_id, "drop_subscription", {"subscription_name": subscription_name})
            logger.info(f"Subscription dropped: {subscription_name} [replication_manager.py:55] [ID:drop_subscription_success]")
            return {"status": "success", "subscription_name": subscription_name}
        except Exception as e:
            error_message = f"Drop subscription failed: {str(e)} [replication_manager.py:60] [ID:drop_subscription_error]"
            logger.error(error_message)
            return {"error": error_message}
