# main/server/mcp/utils/webhook_manager.py
from typing import Dict, Any, List
import redis.asyncio as redis
from fastapi import APIRouter
from ..utils.mcp_error_handler import MCPError
import os
import logging

router = APIRouter()
logger = logging.getLogger("mcp")

class WebhookManager:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.webhooks = {}  # In-memory for simplicity; persist in production

    async def register_webhook(self, user_id: str, endpoint: str, event_types: List[str]) -> str:
        try:
            if not endpoint or not event_types:
                raise MCPError(code=-32602, message="Endpoint and event types are required")
            webhook_id = secrets.token_hex(16)
            self.webhooks[webhook_id] = {
                "user_id": user_id,
                "endpoint": endpoint,
                "event_types": event_types
            }
            await self.redis_client.set(f"webhook:{webhook_id}", json.dumps(self.webhooks[webhook_id]))
            logger.info(f"Registered webhook {webhook_id} for user {user_id}")
            return webhook_id
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Webhook registration failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to register webhook: {str(e)}")

    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        try:
            for webhook_id, webhook in self.webhooks.items():
                if event_type in webhook["event_types"]:
                    await self.redis_client.publish(f"event:{event_type}", json.dumps({
                        "webhook_id": webhook_id,
                        "data": data
                    }))
            logger.info(f"Published event {event_type}")
        except Exception as e:
            logger.error(f"Event publishing failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to publish event: {str(e)}")

    async def deregister_webhook(self, webhook_id: str, user_id: str) -> None:
        try:
            if webhook_id not in self.webhooks or self.webhooks[webhook_id]["user_id"] != user_id:
                raise MCPError(code=-32003, message="Webhook not found or access denied")
            del self.webhooks[webhook_id]
            await self.redis_client.delete(f"webhook:{webhook_id}")
            logger.info(f"Deregistered webhook {webhook_id} for user {user_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Webhook deregistration failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to deregister webhook: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()

@router.post("/webhooks/register")
async def register_webhook_endpoint(user_id: str, endpoint: str, event_types: List[str]):
    manager = WebhookManager()
    try:
        result = await manager.register_webhook(user_id, endpoint, event_types)
        return {"webhook_id": result}
    finally:
        await manager.close()

@router.post("/webhooks/event")
async def publish_event_endpoint(event_type: str, data: Dict[str, Any]):
    manager = WebhookManager()
    try:
        await manager.publish_event(event_type, data)
        return {"status": "success"}
    finally:
        await manager.close()
