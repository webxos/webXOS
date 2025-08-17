from config.config import DatabaseConfig
from lib.security import SecurityHandler
from lib.data_api import DataApiClient
import logging

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.data_api = DataApiClient(db)
        self.project_id = db.project_id

    async def execute_sql(self, user_id: str, query: str, token: str, project_id: str) -> dict:
        try:
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [query_engine.py:15] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            decoded = await self.security.verify_jwt(token)
            if "error" in decoded:
                error_message = f"JWT verification failed: {decoded['error']} [query_engine.py:20] [ID:jwt_error]"
                logger.error(error_message)
                return {"error": error_message}
            result = await self.db.query(query)
            await self.security.log_action(user_id, "sql_query", {"query": query[:50]})
            logger.info(f"SQL query executed: {query[:50]}... [query_engine.py:25] [ID:sql_query_success]")
            return {"status": "success", "data": [dict(row) for row in result]}
        except Exception as e:
            error_message = f"SQL query failed: {str(e)} [query_engine.py:30] [ID:sql_query_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "sql_query", error_message)
            return {"error": error_message}

    async def execute_data_api_query(self, user_id: str, table: str, action: str, token: str, project_id: str, headers: dict, params: dict) -> dict:
        try:
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [query_engine.py:35] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            result = await self.data_api.execute_query(user_id, table, action, token, headers, params)
            await self.security.log_action(user_id, f"data_api_{action}", {"table": table, "params": params})
            logger.info(f"Data API query executed: {table} {action} [query_engine.py:40] [ID:data_api_query_success]")
            return result
        except Exception as e:
            error_message = f"Data API query failed: {str(e)} [query_engine.py:45] [ID:data_api_query_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, f"data_api_{action}", error_message)
            return {"error": error_message}
