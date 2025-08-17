import json
import secrets
from typing import Dict, Any
from ...config.redis_config import get_redis
from ...utils.logging import log_error, log_info

class ToolHandler:
    def __init__(self):
        self.tools = {
            "get_wallet_info": self.handle_wallet_request,
            "generate_credentials": self.handle_generate_credentials
        }

    async def handle_wallet_request(self, user_id: str, redis) -> Dict[str, Any]:
        try:
            wallet_data = await redis.get(f"wallet:{user_id}")
            if not wallet_data:
                wallet_data = {"user_id": user_id, "balance": 0, "currency": "USD"}
                await redis.set(f"wallet:{user_id}", json.dumps(wallet_data), ex=86400)
            log_info(f"Wallet retrieved for user {user_id}")
            return json.loads(wallet_data) if isinstance(wallet_data, str) else wallet_data
        except Exception as e:
            log_error(f"Wallet request failed for {user_id}: {str(e)}")
            raise

    async def handle_generate_credentials(self, user_id: str, redis) -> Dict[str, str]:
        try:
            credentials = {
                "api_key": secrets.token_hex(16),
                "api_secret": secrets.token_hex(32)
            }
            await redis.set(f"credentials:{user_id}", json.dumps(credentials), ex=86400)
            log_info(f"Credentials generated for user {user_id}")
            return credentials
        except Exception as e:
            log_error(f"Credentials generation failed for {user_id}: {str(e)}")
            raise

    async def list_tools(self) -> list:
        return list(self.tools.keys())

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self.tools:
            log_error(f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")
        return await self.tools[name](**args)
