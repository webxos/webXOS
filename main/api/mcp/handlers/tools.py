from fastapi import HTTPException, Depends
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis
from ..mcp_schemas import MCPTool, WalletResponse, WalletRequest
import json
import uuid

class ToolHandler:
    def __init__(self):
        self.tools = {
            "get_wallet_info": self.get_wallet_tool(),
            "generate_credentials": self.get_credentials_tool()
        }

    def get_wallet_tool(self) -> MCPTool:
        return MCPTool(
            name="get_wallet_info",
            description="Retrieve wallet balance and information",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        )

    def get_credentials_tool(self) -> MCPTool:
        return MCPTool(
            name="generate_credentials",
            description="Generate new API key and secret",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        )

    async def handle_tool(self, name: str, arguments: dict, redis=Depends(get_redis)):
        if name == "get_wallet_info":
            return await self.handle_wallet_request(arguments.get("user_id"), redis)
        elif name == "generate_credentials":
            return await self.handle_generate_credentials(arguments.get("user_id"), redis)
        else:
            log_error(f"Tool {name} not found")
            raise HTTPException(status_code=404, detail=f"Tool {name} not found")

    async def handle_wallet_request(self, user_id: str, redis) -> WalletResponse:
        try:
            cached_wallet = await redis.get(f"wallet:{user_id}")
            if cached_wallet:
                log_info(f"Wallet cache hit for user {user_id}")
                return WalletResponse(**json.loads(cached_wallet))
            
            wallet_data = {
                "balance": 0.0,
                "session_balance": 0.0,
                "wallet_key": str(uuid.uuid4()),
                "address": f"0xWEBXOS{str(uuid.uuid4()).replace('-', '')[:16]}",
                "vial_agent": "VialAgent-001",
                "quantum_state": {"qubits": [], "entanglement": "initialized"},
                "reputation": 0,
                "task_status": "Idle"
            }
            await redis.set(f"wallet:{user_id}", json.dumps(wallet_data), ex=3600)
            log_info(f"Wallet retrieved for user {user_id}")
            return WalletResponse(**wallet_data)
        except Exception as e:
            log_error(f"Wallet retrieval failed for {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Wallet error: {str(e)}")

    async def handle_generate_credentials(self, user_id: str, redis) -> dict:
        try:
            key = str(uuid.uuid4())
            secret = str(uuid.uuid4())
            await redis.set(f"credentials:{user_id}", json.dumps({"key": key, "secret": secret}), ex=86400)
            log_info(f"Credentials generated for user {user_id}")
            return {"key": key, "secret": secret}
        except Exception as e:
            log_error(f"Credentials generation failed for {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Credentials error: {str(e)}")
