from postgrest import AsyncPostgrestClient
from config.config import DatabaseConfig
from lib.security import SecurityHandler
import logging
import aiohttp

logger = logging.getLogger(__name__)

class DataApiClient:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.data_api_url = db.data_api_url
        self.client = AsyncPostgrestClient(self.data_api_url)
        self.security = SecurityHandler(db)

    async def close(self):
        try:
            await self.client.aclose()
            logger.info("Data API client closed [data_api.py:15] [ID:data_api_close_success]")
        except Exception as e:
            logger.error(f"Data API client closure failed: {str(e)} [data_api.py:20] [ID:data_api_close_error]")
            raise

    async def execute_query(self, user_id: str, table: str, action: str, token: str, headers: dict, params: dict) -> dict:
        try:
            decoded = await self.security.verify_jwt(token)
            if "error" in decoded:
                error_message = f"JWT verification failed: {decoded['error']} [data_api.py:25] [ID:jwt_error]"
                logger.error(error_message)
                return {"error": error_message}
            self.client.auth(token)
            if action == "select":
                response = await self.client.from_(table).select(params.get("select", "*")).set_params(params).set_headers(headers).execute()
            elif action == "insert":
                response = await self.client.from_(table).insert(params.get("data")).set_params(params).set_headers(headers).execute()
            elif action == "update":
                response = await self.client.from_(table).update(params.get("data")).eq("id", params.get("id")).set_params(params).set_headers(headers).execute()
            elif action == "delete":
                response = await self.client.from_(table).delete().eq("id", params.get("id")).set_params(params).set_headers(headers).execute()
            else:
                error_message = f"Invalid Data API action: {action} [data_api.py:30] [ID:data_api_action_error]"
                logger.error(error_message)
                return {"error": error_message}
            logger.info(f"Data API query executed: {table} {action} [data_api.py:35] [ID:data_api_success]")
            return {"status": "success", "data": response.data}
        except Exception as e:
            error_message = f"Data API query failed: {str(e)} [data_api.py:40] [ID:data_api_error]"
            logger.error(error_message)
            return {"error": error_message}
