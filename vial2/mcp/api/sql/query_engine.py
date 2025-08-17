import asyncpg
from config.config import DatabaseConfig
from postgrest import AsyncPostgrestClient
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")
        self.project_id = "twilight-art-21036984"

    async def execute_sql(self, user_id: str, query: str, token: str, project_id: str) -> Dict[str, Any]:
        if project_id != self.project_id:
            logger.error(f"Invalid Neon project ID: {project_id} [query_engine.py:15] [ID:project_error]")
            return {"error": "Invalid Neon project ID"}
        try:
            # Execute SQL query
            if query.strip().upper().startswith("SELECT"):
                result = await self.db.query(query)
                rows = [dict(row) for row in result]
                logger.info(f"SQL query executed for user {user_id}: {query} [query_engine.py:20] [ID:sql_success]")
                return {"status": "success", "rows": rows}
            else:
                await self.db.query(query)
                logger.info(f"SQL query executed for user {user_id}: {query} [query_engine.py:25] [ID:sql_success]")
                return {"status": "success", "rows": []}
        except asyncpg.exceptions.PostgresError as e:
            error_message = f"SQL query failed: {str(e)} [query_engine.py:30] [ID:sql_error]"
            logger.error(error_message)
            return {"error": error_message}
        except Exception as e:
            error_message = f"Unexpected error: {str(e)} [query_engine.py:35] [ID:unexpected_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def execute_data_api_query(self, user_id: str, table: str, action: str, token: str, project_id: str) -> Dict[str, Any]:
        if project_id != self.project_id:
            logger.error(f"Invalid Neon project ID: {project_id} [query_engine.py:40] [ID:project_error]")
            return {"error": "Invalid Neon project ID"}
        try:
            self.data_api.auth(token)
            if action == "select":
                response = await self.data_api.from_(table).select("*").eq("user_id", user_id).eq("project_id", project_id).execute()
            elif action == "insert":
                response = await self.data_api.from_(table).insert({"user_id": user_id, "project_id": project_id}).execute()
            elif action == "update":
                response = await self.data_api.from_(table).update({"user_id": user_id}).eq("user_id", user_id).eq("project_id", project_id).execute()
            elif action == "delete":
                response = await self.data_api.from_(table).delete().eq("user_id", user_id).eq("project_id", project_id).execute()
            else:
                error_message = f"Invalid Data API action: {action} [query_engine.py:50] [ID:action_error]"
                logger.error(error_message)
                return {"error": error_message}
            logger.info(f"Data API query executed for user {user_id}: {table} {action} [query_engine.py:55] [ID:data_success]")
            return {"status": "success", "rows": response.data}
        except Exception as e:
            error_message = f"Data API query failed: {str(e)} [query_engine.py:60] [ID:data_error]"
            logger.error(error_message)
            return {"error": error_message}
